document.getElementById('analyzeForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const ticker = document.getElementById('ticker').value.trim();
  const prompt_choice = document.getElementById('prompt_choice').value;
  const fd = new FormData();
  fd.append('ticker', ticker);
  fd.append('prompt_choice', prompt_choice);

  const resDiv = document.getElementById('result');
  resDiv.innerHTML = 'Analizando...';

  try {
    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) {
      resDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
      return;
    }
    let html = `<h2>${data.ticker} — Precio actual: ${data.precio_actual}</h2>`;
    html += `<p>Señal (ejemplo): <strong>${data.senal}</strong></p>`;
    html += `<h3>Predicciones</h3>`;
    html += `<p>Regresión 7d: ${data.regresion_pred_7d}</p>`;
    html += `<p>LSTM 7d: ${data.lstm_pred_7d}</p>`;
    html += `<h3>Sentimiento (muestra)</h3>`;
    html += `<pre>${JSON.stringify(data.sentimiento, null, 2)}</pre>`;
    html += `<h3>Noticias (muestra)</h3>`;
    if (data.noticias && data.noticias.length) {
      html += '<ul>';
      data.noticias.forEach(n => {
        html += `<li><a href="${n.url}" target="_blank">${n.title || n.source}</a></li>`;
      });
      html += '</ul>';
    }
    if (data.chart_png_b64) {
      html += `<h3>Gráfico</h3><img src="data:image/png;base64,${data.chart_png_b64}" style="max-width:100%;">`;
    }
    resDiv.innerHTML = html;
  } catch (err) {
    document.getElementById('result').innerText = 'Error: ' + err.message;
  }
});
