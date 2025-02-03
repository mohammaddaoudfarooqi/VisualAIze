function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('collapsed');
}

function showIframe(iframeId) {
    document.getElementById('iframe1').style.display = 'none';
    document.getElementById('iframe2').style.display = 'none';
    document.getElementById(iframeId).style.display = 'block';
}


async function navigateToPage(url) {
    const response = await fetch(url);
    const html = await response.text();
    document.getElementById("content").innerHTML = html;
}