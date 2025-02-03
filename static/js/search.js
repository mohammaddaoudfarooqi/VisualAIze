document.addEventListener("DOMContentLoaded", function () {
  const searchInput = document.getElementById("search-input");
  const resultsContainer = document.getElementById("results");
  const llmResponse_Container = document.getElementById("llmResponse");
  const autocompleteList = document.getElementById("autocomplete-list");

  // Handle search input and fetch suggestions
  searchInput.addEventListener("input", async function () {
    const query = searchInput.value;
    try {
      const response = await fetch(
        `/autocomplete?query=${encodeURIComponent(query)}`
      );
      const data = await response.json();

      autocompleteList.innerHTML = "";
      data.suggestions.forEach((suggestion) => {
        const item = document.createElement("div");
        item.textContent = suggestion;
        item.className = "autocomplete-item";
        item.addEventListener("click", () => {
          searchInput.value = suggestion;
          autocompleteList.innerHTML = "";
          fetchSearchResultsAndAskLLM();
        });
        autocompleteList.appendChild(item);
      });
    } catch (error) {
      console.error("Error fetching autocomplete suggestions:", error);
    }
  });

  // Handle search query and fetch results
  async function fetchSearchResults() {
    const query = searchInput.value;
    const selectedCategories = Array.from(
      document.querySelectorAll("#category-filter input:checked")
    )
      .map((input) => encodeURIComponent(input.value)) // Encode each category
      .join(",");
    llmResponse_Container.innerHTML = "";

    try {
      // Fetch search results
      const searchResponse = await fetch(
        `/search?query=${encodeURIComponent(
          query
        )}&categories=${selectedCategories}`
      );
      const searchData = await searchResponse.json();

      resultsContainer.innerHTML = "";
      // Loop through search results and create individual cards
      searchData.results.forEach((result) => {
        const card = document.createElement("div");
        card.className = "result-card";
        card.innerHTML = `
          <a href="${result.url}" target="_blank">    
            <h5>${result.category}</h5>
            <img src="${result.image}" alt="${result.title}">
            <h3>${result.title}</h3>
            <p>${result.description}</p>
            <p>Read more ></p>
          </a>
        `;
        resultsContainer.appendChild(card);
      });
    } catch (error) {
      console.error("Error fetching search results or llm response:", error);
    }
  }

  async function fetchSearchResultsAndAskLLM() {
    const query = searchInput.value;
    const selectedCategories = Array.from(
      document.querySelectorAll("#category-filter input:checked")
    )
      .map((input) => encodeURIComponent(input.value)) // Encode each category
      .join(",");

    try {
      // Fetch search results
      const searchResponse = await fetch(
        `/search?query=${encodeURIComponent(
          query
        )}&categories=${selectedCategories}`
      );
      const searchData = await searchResponse.json();

      resultsContainer.innerHTML = "";
      llmResponse_Container.innerHTML = "";
      // Loop through search results and create individual cards
      searchData.results.forEach((result) => {
        const card = document.createElement("div");
        card.className = "result-card";
        card.innerHTML = `
          <a href="${result.url}" target="_blank">    
            <h5>${result.category}</h5>
            <img src="${result.image}" alt="${result.title}">
            <h3>${result.title}</h3>
            <p>${result.description}</p>
            <p>Read more ></p>
          </a>
        `;
        resultsContainer.appendChild(card);
      });

      // Fetch LLM response
      const llmResponse = await fetch(
        `/llm_response?query=${encodeURIComponent(query)}`
      );
      const llmData = await llmResponse.json();

      // Add a single LLM response summary at the top, if available
      if (llmData.llmResponse) {
        const llmResponseContainer = document.createElement("div");
        llmResponseContainer.className = "llm-summary";
        llmResponseContainer.innerHTML = `<p><b>LLM Response:</b><br/><br/>${llmData.llmResponse}</p>`;
        llmResponse_Container.innerHTML = "";
        llmResponse_Container.appendChild(llmResponseContainer);
      } else {
        llmResponse_Container.innerHTML = "";
      }
    } catch (error) {
      console.error("Error fetching search results or llm response:", error);
    }
  }

  // Trigger search on category change
  document
    .getElementById("category-filter")
    .addEventListener("change", fetchSearchResults);
  searchInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      autocompleteList.innerHTML = "";
      fetchSearchResultsAndAskLLM();
    }
    if (e.key === "Escape") {
      autocompleteList.innerHTML = "";
    }
  });

  // Call fetchSearchResults on page load to load results immediately
  fetchSearchResults();
});
