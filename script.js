const selector = document.getElementById("csvSelector");
const downloadLink = document.getElementById("downloadLink");

selector.addEventListener("change", function () {
    const selectedFile = selector.value;
    downloadLink.href = selectedFile;
});
