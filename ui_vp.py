import asyncio
import base64
import json
import os
import re
import time
import uuid
from textwrap import wrap

import aiohttp
import boto3
import cv2
import dotenv
import gradio as gr
import pysrt
from gradio import Markdown as m
from langchain_aws import BedrockEmbeddings
from pymongo import MongoClient
from skimage.metrics import structural_similarity as compare_ssim

dotenv.load_dotenv()

# Configuration Constants from .env
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
MONGODB_URI = os.getenv("MONGODB_URI")


def generate_unique_filename(prefix: str = "", extension: str = "") -> str:
    """
    Generates a unique filename for S3 using a UUID without dashes and optional prefix and extension.

    :param prefix: (str) Optional prefix for the filename (e.g., folder name).
    :param extension: (str) File extension, including the leading dot (e.g., ".txt").
    :return: (str) A unique filename string.
    """
    # Generate a UUID and remove dashes
    unique_id = uuid.uuid4().hex  # .hex generates a string without dashes
    
    # Add a timestamp for extra uniqueness (optional)
    timestamp = int(time.time())
    
    # Construct the filename
    filename = f"{prefix}{unique_id}_{timestamp}"
    
    # Add file extension if provided
    if extension:
        filename += extension
    
    return filename



# Step 1: Upload video to S3
async def upload_to_s3(file_path, file_name):
    """Asynchronously upload a file to an S3 bucket."""
    try:
        s3_client = boto3.client("s3")
        loop = asyncio.get_event_loop()
        # Run the upload operation in a thread to keep it async
        await loop.run_in_executor(None, s3_client.upload_file, file_path, BUCKET_NAME, file_name)
        s3_url = f"s3://{BUCKET_NAME}/{file_name}"
        return f"File uploaded to S3 successfully! You may access it here: {s3_url}"
    except Exception as e:
        return f"An error occurred: {str(e)}"



# Step 2: Start Transcription Job
async def start_transcription_job(bucket_name, file_name):
    """Start a transcription job in AWS Transcribe."""
    transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
    job_uri = f"s3://{bucket_name}/{file_name}"
    transcribe_job_name = f"VideoTranscriptionJob-{int(time.time())}"
    transcribe_client.start_transcription_job(
        TranscriptionJobName=transcribe_job_name,
        Media={"MediaFileUri": job_uri},
        MediaFormat="mp4",
        LanguageCode="en-US"
    )
    return transcribe_job_name

# Step 3: Wait for Transcription Job to Complete
async def wait_for_transcription(transcribe_job_name):
    """Wait asynchronously for transcription job to complete."""
    transcribe_client = boto3.client("transcribe", region_name=AWS_REGION)
    while True:
        status = transcribe_client.get_transcription_job(
            TranscriptionJobName=transcribe_job_name
        )
        if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
            break
        print("Waiting for transcription to complete...")
        await asyncio.sleep(10)
    if status["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
        transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        return transcript_uri
    else:
        raise Exception("Transcription failed!")


async def fetch_transcript(transcript_url, output_filename):
    """Fetch the transcript from the provided URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(transcript_url) as response:
            if response.status == 200:
                transcript_text = await response.text()
                with open(output_filename, "w") as file:
                    file.write(transcript_text)
                return "Transcript downloaded successfully."
            else:
                raise Exception(f"Failed to download transcript. Status Code: {response.status}")



async def summarize_with_claude(transcript,summary_filename, model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"):
    """Summarizes the transcript using Claude via AWS Bedrock."""
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    # Define the prompt for the model.
    prompt = (
    "You are a highly capable AI assistant specialized in processing transcripts and generating "
    "concise, structured summaries. I am providing you with a transcript. Your task is to summarize "
    "the transcript clearly and concisely. The summary should have the following characteristics:\n"
    "- Focus on the main topics, key points, and notable statements or actions.\n"
    "- Provide an overarching theme or purpose of the transcript.\n"
    "- Highlight specific insights, decisions, or resolutions made, if applicable.\n"
    "- Maintain neutrality and avoid personal interpretation.\n"
    "- Organize the summary into logical sections.\n"
    "- Use bullet points or paragraphs as appropriate for clarity.\n\n"
    "The output should follow this format:\n\n"
    "**Title:** A clear, brief title summarizing the transcript topic.\n"
    "**Summary:**\n"
    "    - **Context:** One to two sentences about the setting or purpose of the transcript.\n"
    "    - **Key Discussion Points:** List or describe the major topics covered.\n"
    "    - **Important Outcomes/Actions:** Any conclusions, decisions, or next steps discussed.\n"
    "    - **Notable Statements/Quotes:** Include specific statements if relevant.\n\n"
    "The transcript is as follows:\n\n"
    f"{transcript}"
)

    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "temperature": 0.7,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: client.invoke_model(modelId=model_id, body=request)
    )
    model_response = json.loads(response["body"].read())
    response_text = model_response["content"][0]["text"]
    with open(summary_filename, "w") as file:
        file.write(response_text)
    return response_text


async def generate_srt_file(transcript_data, output_file="subtitles.srt", line_width=42):
    """
    Generates an SRT file using audio_segments and items, breaking on full stops, 
    wrapping text lines, and correcting spelling mistakes.
    """
    #spell = SpellChecker()  # Initialize the spell checker
    audio_segments = transcript_data["results"]["audio_segments"]
    items = transcript_data["results"]["items"]
    id=1

    srt_entries = []
    punctuation_marks = {".", "!", "?",","}  # Supported punctuation marks to break sentences

    for segment in audio_segments:
        segment_items = segment["items"]
        sentence = ""
        start_time = None

        for idx in segment_items:
            item = items[idx]

            if item["type"] == "pronunciation":
                # Append the word to the sentence
                word = item["alternatives"][0]["content"]
                #corrected_word = spell.correction(word)  # Correct spelling
                if not start_time:
                    start_time = float(item["start_time"])  # Start time of the sentence
                sentence += word + " "

            elif item["type"] == "punctuation":
                punctuation = item["alternatives"][0]["content"]
                sentence = sentence.strip() + punctuation + " "

                if punctuation in punctuation_marks:
                    # End sentence and create an SRT entry
                    previous_item = items[segment_items[segment_items.index(idx) - 1]]  # Last pronunciation
                    end_time = float(previous_item["end_time"])  # Get end time from the last word

                    # Format timestamps for SRT
                    start_time_srt = format_timestamp(start_time)
                    end_time_srt = format_timestamp(end_time)

                    # Wrap text and add to SRT entry
                    wrapped_sentence = "\n".join(wrap(sentence.strip(), width=line_width))
                    srt_entry = f"\n{id}\n{start_time_srt} --> {end_time_srt}\n{wrapped_sentence}\n"
                    srt_entries.append(srt_entry)
                    id+=1

                    # Reset sentence and start time
                    sentence = ""
                    start_time = None

        # Handle leftover sentence (no final punctuation)
        if sentence:
            last_item = items[segment_items[-1]]  # Last item in the segment
            end_time = float(last_item.get("end_time", start_time))  # Use start_time if end_time is missing
            start_time_srt = format_timestamp(start_time)
            end_time_srt = format_timestamp(end_time)

            # Wrap text and add to SRT entry
            wrapped_sentence = "\n".join(wrap(sentence.strip(), width=line_width))
            srt_entry = f"\n{id}\n{start_time_srt} --> {end_time_srt}\n{wrapped_sentence}\n"
            srt_entries.append(srt_entry)
            id+=1

    # Write to SRT file
    with open(output_file, "w") as file:
        file.writelines(srt_entries)

    print(f"Subtitles written to {output_file}")
    return srt_entries


def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"



#region videoanalysis
bedrock_embeddings = BedrockEmbeddings(
    region_name="us-east-1", model_id="amazon.titan-embed-text-v1"
)
MONGODB_URI = os.getenv("MONGODB_URI")
# Initialize AWS Bedrock and MongoDB
bedrock_client = boto3.client(
    "bedrock-runtime", region_name="us-east-1"
)  # Update with correct region
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["video_analysis"]
collection = db["frames_data"]

# Convert timestamp from SRT to milliseconds
def timestamp_to_milliseconds(timestamp):
    return (
        timestamp.hours * 3600 + timestamp.minutes * 60 + timestamp.seconds
    ) * 1000 + timestamp.milliseconds


# Calculate the middle of a timestamp range
def calculate_midpoint(start, end):
    start_ms = timestamp_to_milliseconds(start)
    end_ms = timestamp_to_milliseconds(end)
    midpoint_ms = (start_ms + end_ms) // 2
    return midpoint_ms / 1000.0  # Convert milliseconds to seconds


# Extract a specific frame from the video
def extract_frame(video_path, timestamp_seconds):
    cap = cv2.VideoCapture(video_path)
    cap.set(
        cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000
    )  # Set position in milliseconds
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None


# Compare two frames for uniqueness
def is_unique_frame(new_frame, previous_frame, threshold=0.5):
    if previous_frame is None:
        return True  # First frame is always unique

    # Convert frames to grayscale
    gray_new = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Compute SSIM (Structural Similarity Index)
    score, _ = compare_ssim(gray_new, gray_prev, full=True)
    return score < threshold  # Lower SSIM indicates more significant change


# Analyze frame with Claude via Bedrock
async def analyze_frame_with_claude(
    image_path,
    transcript_summary,
    spoken_text,
    transcript,
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    prompt = (
        "You are a highly capable AI assistant with perfect vision and exceptional attention to detail, "
        "specialized in analyzing images and extracting comprehensive information. I am providing you with "
        "an image that is a frame from a video. The transcript summary of the video, the transcript and text spoken in the video on this frame has already been given. "
        "Your task is to analyze the image meticulously and extract all possible details. Ensure that your "
        "analysis is thorough and considers every aspect of the image. Before providing your answer, think "
        "step by step and analyze every part of the image systematically.\n\n"
        "Your analysis should include the following:\n"
        "- Identify the primary subject(s) of the image and their attributes (e.g., appearance, clothing, expressions).\n"
        "- Describe the environment or setting, including background details and any notable objects.\n"
        "- Note any actions or interactions occurring in the image.\n"
        "- Highlight any text, symbols, or signage visible in the image.\n"
        "- Consider the lighting, color scheme, and mood conveyed by the image.\n"
        "- Relate the details of the image to the provided transcript.\n"
        "- Maintain a logical and structured approach in your analysis.Give direct answers.\n\n"
        "- Structure your reponse for better semantic search retrieval.\n\n"
        "- Refer to the text from the image to get correct spellings of texts in the transcript.\n\n"
        "- Avoid providing preamble such as 'Here is the answer','Here is a detailed analysis of the image','Certainly, let me provide a detailed analysis of the image', etc"
        "The text spoken in the video on this frame is as follows:\n\n"
        f"{spoken_text}\n\n"
        "The transcript summary is as follows:\n\n"
        f"{transcript_summary}\n\n"
        "The transcript is as follows:\n\n"
        f"{transcript}\n\n"
    )

    prompt = (
        "You are an advanced AI assistant with exceptional attention to detail, specialized in analyzing images and extracting comprehensive information "
        "from frames in a video. The video provides a specific context, and each frame reflects key elements of the subject matter discussed. "
        "I will provide you with a frame from the video along with its transcript and spoken text. Your task is to analyze the image in detail, "
        "extracting all relevant elements that reflect the content and context described in the transcript. Be systematic, step-by-step, in your approach to ensuring no detail is overlooked.\n\n"
        "Your analysis should be highly detailed and structured, optimized for future retrieval from MongoDB database. Focus on the following key areas:\n\n"
        "1. **Primary Subject(s):** Identify the main subject(s) in the image (e.g., people, objects, or scenes). "
        "Provide a detailed description of their appearance, attire, expressions, and any notable features. For example, if a person is visible, describe their clothing, posture, and any interactions occurring. If objects or scenes are present, describe them in detail.\n"
        "   - Example format: 'subject_id': 'subject_001', 'description': 'Person wearing a blue suit with a microphone...', 'attributes': {...}\n"
        "2. **Setting & Background:** Describe the environment shown in the frame, including any visible objects, background elements, or locations. "
        "If the image reflects a particular setting, be it indoors, outdoors, or any other relevant environment, describe it clearly. Mention relevant objects and their context.\n"
        "   - Example format: 'description': 'A conference room with a projector displaying a presentation...', 'objects': ['projector', 'presentation slide', 'whiteboard']\n"
        "3. **Actions & Interactions:** Identify any actions taking place in the frame, including physical actions, interactions with objects, or gestures. "
        "If a person is performing an action (e.g., talking, pointing, writing), describe it. If objects are interacting, such as technology being used, explain what is happening in the scene.\n"
        "   - Example format: 'action': 'Person presenting a slide deck while interacting with a digital screen.', 'subjects_involved': ['subject_001']\n"
        "4. **Text & Symbols:** Transcribe any visible text in the image, including UI elements, labels, signage, or any on-screen text that provides context. "
        "Be sure to capture the text exactly as it appears, noting its location in the image (e.g., top-left, bottom-right), and its significance to the overall scene.\n"
        "   - Example format: 'text': 'Welcome to the presentation', 'location': 'Top left corner of the screen', 'context': 'Introduction screen for a business presentation.'\n"
        "5. **Visual Style & Mood:** Describe the lighting, color scheme, and mood of the frame. For example, if the image conveys a formal, casual, or relaxed mood, mention this. "
        "Also, describe the use of colors, shadows, and highlights, and how they influence the overall tone of the image.\n"
        "   - Example format: 'lighting': 'Soft, with a warm color temperature', 'color_palette': 'Light blues and grays', 'mood': 'Professional and calm'\n"
        "6. **Transcript Relation:** Relate the image details to the provided transcript. How does the image reflect or represent the content being discussed in the transcript? "
        "For example, if the speaker is discussing a specific concept, does the image show visuals related to that concept? Provide any necessary context from the spoken text to make these connections.\n"
        "   - Example format: 'spoken_text': 'This solution provides advanced data analytics...', 'transcript_summary': 'Explaining the capabilities of a new data analytics tool.', 'full_transcript': 'Full text from the transcript.'\n"
        "7. **Structured Analysis:** Include any other details that help with retrieving and understanding the analysis later. This can include keywords, relevant concepts, or themes from the video. "
        "Ensure the analysis is structured to make it easy for semantic search, highlighting keywords like 'data analytics', 'presentation', 'innovation', or 'solution'.\n"
        "   - Example format: 'logical_structure': 'The analysis organizes the elements by subject, background, actions, and text for easy retrieval.', 'semantic_search_keywords': ['data analytics', 'business presentation', 'innovation', 'solution']\n"
        "Your response should be structured in JSON format for optimal storage in MongoDB database, with each part clearly defined to ensure efficient and accurate retrieval. OUTPUT ONLY A VALID JSON AND NOTHING ELSE! "
        "DO NOT include introductory phrases like 'Here is the answer,' 'Let me analyze the image,','Here is the detailed analysis of the image in JSON format', OR ANYTHING IN SIMILAR LINES.\n"
        "Be as thorough and descriptive as possible to fully capture the scene and make it easy to link the image to the context of the video.\n\n"
        """OUTPUT JSON FORMAT:
    {
    "primary_subjects": [
        {
            "subject_id": "unique_subject_id",
            "description": "Detailed description of the subject(s) including appearance, clothing, expressions, and attributes, only if available.Do not guess.",
            "attributes": {
                "clothing": "clothing description",
                "expression": "expression description",
                "distinctive_features": "any distinguishing features"
            }
        }
    ],
    "setting_and_background": {
        "description": "Detailed description of the environment or setting, including background details and any notable objects. Do not guess.",
        "objects": [
            "object_1",
            "object_2"
        ]
    },
    "actions_and_interactions": [
        {
            "action": "Description of the action or interaction happening in the image",
            "subjects_involved": ["subject_id_1", "subject_id_2"]
        }
    ],
    "text_and_symbols": [
        {
            "text": "Visible text or signage in the image",
            "location": "Location or position in the image",
            "context": "Context or relevance of the text"
        }
    ],
    "visual_style_and_mood": {
        "lighting": "Description of lighting",
        "color_palette": "Color scheme used",
        "mood": "Mood or emotional tone conveyed by the image"
    },
    "transcript_relation": {
        "spoken_text": "Exact spoken text from the video.Spell check using the texts from the image.",
        "transcript_summary": "Summary of the transcript.Spell check using the texts from the image.",
        "full_transcript": "Full transcript of the video. Spell check the transcript using the texts from the image.".
    },
    "structured_analysis": {
        "logical_structure": "Explanation of the overall analysis process and how it is structured for easy understanding and retrieval",
        "semantic_search_keywords": [
            "keyword_1",
            "keyword_2",
            "keyword_3"
        ]
    },
    "questions":["question_1","question_2",..] create a list of well formed relevant questions that could be asked using only the extracted details. 
}

    """
        "The spoken text from the video on this frame is as follows:\n\n"
        f"{spoken_text}\n\n"
        "The transcript summary is as follows:\n\n"
        f"{transcript_summary}\n\n"
        "The full transcript is as follows:\n\n"
        f"{transcript}\n\n"
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 8192,
        "anthropic_version": "bedrock-2023-05-31",
    }

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: bedrock_client.invoke_model(modelId=model_id,contentType="application/json", body=json.dumps(payload))
    )

    output_binary = response["body"].read()
    output_json = json.loads(output_binary)
    output = output_json["content"][0]["text"]
    return output


# Store analysis results in MongoDB
async def store_analysis_data(id, timestamp, title, frame_name, analysis_result, spoken_text):
    analysis_result = json.loads(analysis_result)

    data={
            "title": title,
            "spoken_text": spoken_text,
            "chunk_summary":str(analysis_result["transcript_relation"]["transcript_summary"]),
            "llm_analysis_result": analysis_result
        }
    document_text = str(data)
    document_embedding = bedrock_embeddings.embed_query(document_text)
    insert_data={
            "sequence_id": id,
            "timestamp": json.loads(json.dumps(timestamp)),
            "title": title,
            "spoken_text": spoken_text,
            "frame_file_name": frame_name,
            "llm_analysis_result": analysis_result,
            "document_text":document_text,
            "document_embedding":document_embedding

        }
    collection.insert_one(
        insert_data
    )

    return(f"Stored analysis for frame {frame_name} in MongoDB.")


# Process video with SRT, extracting unique frames
async def process_video_with_unique_frames(
    video_path, srt_path,summary_path,transcript_path,output_folder, uniqueness_threshold=0.9
):
    subtitles = pysrt.open(srt_path)
    previous_frame = None
    with open(summary_path, "r") as file:
        summary = file.read()
        transcript = ""

    title_match = re.search(r"\*\*Title:\*\* (.+)", summary)

    # Check if a match was found
    if title_match:
        title = title_match.group(1)
    # Step 2: Read transcript file
    with open(transcript_path, "r") as file:
        transcript_data = json.load(file)
        transcript = transcript_data["results"]["transcripts"][0]["transcript"]
    id = 1
    spoken_text = ""
    previous_start_time = None
    previous_end_time = None
    last = 0

    for i, subtitle in enumerate(subtitles):
        last = i
        start_time = subtitle.start
        end_time = subtitle.end

        midpoint_seconds = calculate_midpoint(start_time, end_time)

        # Extract frame at midpoint
        frame = extract_frame(video_path, midpoint_seconds)

        if frame is not None and is_unique_frame(
            frame, previous_frame, uniqueness_threshold
        ):
            if previous_start_time is None:
                previous_start_time = (
                    start_time  # Set the start time if not already set
                )

            previous_end_time = end_time  # Update end time to the latest
            spoken_text += subtitle.text
            frame_name = f"frame_{i+1}"
            frame_path = os.path.join(output_folder, f"{frame_name}.jpg")

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            cv2.imwrite(frame_path, frame)
            print(f"Saved unique frame: {frame_name}")

            # Analyze and store in MongoDB
            analysis_result =await analyze_frame_with_claude(
                frame_path, summary, spoken_text, transcript
            )
            timestamp = {
                "start_time": str(previous_start_time),
                "end_time": str(previous_end_time),
            }
            await store_analysis_data(
                id, timestamp, title, frame_name, analysis_result, spoken_text
            )
            id += 1

            # Update previous frame
            previous_frame = frame
            spoken_text = ""
            previous_start_time = None  # Reset start time for next sequence
            previous_end_time = None  # Reset end time for next sequence
        else:
            print(f"Frame at {midpoint_seconds} seconds is not unique, skipping.")
            spoken_text += subtitle.text
            if previous_start_time is None:
                previous_start_time = start_time  # Keep the earliest start time
            previous_end_time = end_time  # Always update to the latest end time

    # Ensure the last frame is included
    if (
        spoken_text
        and previous_start_time is not None
        and previous_end_time is not None
    ):
        frame_name = f"frame_{last+1}.jpg"
        frame_path = os.path.join(output_folder, f"{frame_name}.jpg")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if previous_frame is not None:
            cv2.imwrite(frame_path, previous_frame)
            print(f"Saved last frame: {frame_name}")

        # Analyze and store in MongoDB
        analysis_result = await analyze_frame_with_claude(
            frame_path, summary, spoken_text, transcript
        )
        timestamp = {
            "start_time": str(previous_start_time),
            "end_time": str(previous_end_time),
        }
        await store_analysis_data(
            id, timestamp, title, frame_name, analysis_result, spoken_text
        )
    return

#endregion



async def handle_upload(file_path):
    """Handle the file upload and save it to S3."""
    progress=""
    if file_path is None:
        progress+="No file uploaded."
        yield [progress,"","",""]
        return

    file_name = os.path.basename(file_path)
    name, extension = os.path.splitext(file_name)
    unique_filename = generate_unique_filename(prefix="outputs/"+name+"_", extension=extension)
    progress+="\nUploading File: "+unique_filename+" to AWS S3."
    yield [progress,"","",""]

    # Upload to S3
    progress+="\n"+ await upload_to_s3(file_path, unique_filename)
    yield [progress,"","",""]

    progress+="\nTranscribing..."
    yield [progress,"","",""]
    job_name = await start_transcription_job(BUCKET_NAME, unique_filename)
    transcript_uri = await wait_for_transcription(job_name)
    progress+="\nTranscription URI: "+transcript_uri
    yield [progress,"","",""]


    #Process video: fetch transcript, summarize, and generate subtitles.
    # Step 1: Fetch transcript
    transcript_filename = generate_unique_filename(prefix="outputs/"+name+"_Transcript_", extension=".json")
    fetch_result = await fetch_transcript(transcript_uri, transcript_filename)
    progress += f"\n{fetch_result} - {transcript_filename}"
    yield [progress,"","",""]

    transcript=""
    # Step 2: Read transcript file
    with open(transcript_filename, "r") as file:
        transcript_data = json.load(file)
        transcript = transcript_data["results"]["transcripts"][0]["transcript"]
        yield [progress,transcript,"",""]


    # Step 3: Generate SRT file
    progress+="\nGenerating subtitles..."
    yield [progress,transcript,"",""]
    subtitles_filename = generate_unique_filename(prefix="outputs/"+name+"_Subtitles_", extension=".srt")
    
    subtitles = await generate_srt_file(transcript_data,output_file=subtitles_filename)
    subtitles="".join(subtitles)
    progress+="\nSubtitles at: "+subtitles_filename
    yield [progress,transcript,subtitles,""]

    # Step 4: Summarize using Claude
    summary_filename = generate_unique_filename(prefix="outputs/"+name+"_Summary_", extension=".txt")
    progress+="\nSummarizing with Claude..."
    yield [progress,transcript,subtitles,""]
    summary= await summarize_with_claude(transcript,summary_filename)
    progress+="\nSummary at: "+summary_filename
    yield [progress,transcript,subtitles,summary]

    output_folder = "outputs/extracted_frames"
    progress+="\nVisualAIze is watching and analyzing the video...Please wait.... "
    yield [progress,transcript,subtitles,summary]
    await process_video_with_unique_frames(file_path, subtitles_filename,summary_filename,transcript_filename,output_folder)
    progress+="\nAnalysis Complete! You may ask questions on the video now!"
    yield [progress,transcript,subtitles,summary]


custom_css = """
            footer {visibility: hidden}; 
        """

with gr.Blocks(
    head="",
    fill_height=True,
    fill_width=True,
    css=custom_css,
    title="VisualAIze",
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.green),
) as demo:
    m("""<center><h1>Upload Video!</h1></center><br/>""")
    with gr.Row():
        videoUserInput = gr.Video(label="Upload a Video",container=True,)
        
    with gr.Row():
        submit_button = gr.Button("Submit", render=True)
    with gr.Row():
        txtProgress = gr.Textbox(
                label='Progress',
                show_label=True,
                container=True,
                autoscroll=True,
            )
    with gr.Row():
        txtTranscript = gr.Textbox(
                label='Transcript',
                show_label=True,
                container=True,
                autoscroll=True,
                show_copy_button=True
            )
        txtSubtitle = gr.Textbox(
                label='Subtitle',
                show_label=True,
                container=True,
                autoscroll=True,
                show_copy_button=True
            )
    with gr.Row():
        txtSummary = gr.Textbox(
                label='Summary',
                show_label=True,
                container=True,
                autoscroll=True,
                show_copy_button=True
            )
        outputs = [txtProgress,txtTranscript,txtSubtitle,txtSummary]
    

    # Link the submit button to the async function with Gradio's async support
    submit_button.click(
        fn=handle_upload,
        inputs=videoUserInput,
        outputs=outputs,
    )
