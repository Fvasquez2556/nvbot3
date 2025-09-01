// Scripts para NvBot3 Dashboard
function refreshData() {
    console.log('Refrescando datos...');
    location.reload();
}

function submitFeedback(signalId, feedbackType) {
    const data = {
        signal_id: signalId,
        feedback_type: feedbackType,
        comments: prompt('Comentarios adicionales (opcional):') || ''
    };
    
    fetch('/api/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('¡Feedback guardado correctamente!');
            location.reload();
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error enviando feedback');
    });
}

// Auto-refresh cada 30 segundos
setInterval(refreshData, 30000);
