document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();                         // keep page from reloading

    const file = document.getElementById("fileInput").files[0];
    if (!file) return;

    const fd = new FormData();
    fd.append("image", file);                   // key must be "image"

    const resultBox = document.getElementById("result");
    resultBox.textContent = "Classifying...";

    try {
        const resp = await fetch("/api/classify", { method: "POST", body: fd });
        const data = await resp.json();

        if (resp.ok) {
            resultBox.textContent = data.predictions
                .map(([label, prob], i) =>
                    (i + 1) + ". " + label + " - " + (prob * 100).toFixed(2) + "%")
                .join("\n");
        } else {
            resultBox.textContent =
                "Error: " + (data.error || resp.statusText);
        }
    } catch (err) {
        resultBox.textContent = "Error: " + err.message;
    }
});
