document.getElementById('threshold').addEventListener('input', function() {
    document.getElementById('threshold-value').textContent = this.value;
});

const preview = document.getElementById('preview');
if (preview) {
    preview.addEventListener('click', function(e) {
        const rect = this.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const imgX = Math.round((x / rect.width) * this.naturalWidth);
        const imgY = Math.round((y / rect.height) * this.naturalHeight);
        
        document.getElementById('click-coords').textContent = `${imgX}, ${imgY}`;
        
        fetch('/process_click', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_url: "{{ image_url }}",
                x: imgX,
                y: imgY,
                threshold: document.getElementById('threshold').value
            })
        })
        .then(response => response.json())
        .then(data => {
            preview.src = data.result_url + '?t=' + new Date().getTime();
        });
    });
}