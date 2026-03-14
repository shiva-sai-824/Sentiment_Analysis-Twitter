const COLORS = {
  positive: { stroke: '#4f6de5', bar: 'positive-active' },
  neutral:  { stroke: '#e8622a', bar: 'neutral-active'  },
  negative: { stroke: '#888',    bar: 'negative-active' },
};

function setActive(type) {
  ['positive', 'neutral', 'negative'].forEach(t => {
    const svg   = document.getElementById('robot-' + t);
    const bar   = document.getElementById('bar-'   + t);
    const lbl   = document.getElementById('label-' + t);
    const c     = t === type ? COLORS[type].stroke : '#9aa5c4';
    const scale = t === type ? 1.15 : 0.9;

    svg.style.transform = `scale(${scale})`;

    svg.querySelectorAll('rect, circle, path, polygon').forEach(el => {
      if (el.hasAttribute('stroke') && el.getAttribute('stroke') !== 'none')
        el.setAttribute('stroke', c);
      if (el.hasAttribute('fill') && el.getAttribute('fill') !== 'none')
        el.setAttribute('fill', c);
    });

    bar.className = 'robot-bar' + (t === type ? ' ' + COLORS[type].bar : '');
    lbl.className = 'robot-label-top' + (t === type ? ' active' : '');
  });
}

function resetRobots() {
  ['positive', 'neutral', 'negative'].forEach(t => {
    const svg = document.getElementById('robot-' + t);
    svg.style.transform = 'scale(1)';
    svg.querySelectorAll('rect, circle, path, polygon').forEach(el => {
      if (el.hasAttribute('stroke') && el.getAttribute('stroke') !== 'none')
        el.setAttribute('stroke', '#9aa5c4');
      if (el.hasAttribute('fill') && el.getAttribute('fill') !== 'none')
        el.setAttribute('fill', '#9aa5c4');
    });
    document.getElementById('bar-'   + t).className = 'robot-bar';
    document.getElementById('label-' + t).className = 'robot-label-top';
  });
}

async function analyzeSentiment() {
  const textInput = document.getElementById('textInput').value.trim();
  const resultEl  = document.getElementById('result');

  if (!textInput) {
    resultEl.innerHTML = '<span class="error">Please provide input text</span>';
    return;
  }

  resultEl.innerHTML = '<span class="loading">Analyzing...</span>';
  resetRobots();

  try {
    const response = await fetch('https://shivasai824-twitter-sentiment-api.hf.space/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: textInput }),
    });

    if (!response.ok) throw new Error('Network error');
    const data = await response.json();
    const sentiment = data.sentiment || '';

    // Check if sarcasm was detected
    const isSarcastic = sentiment.includes('Sarcasm Detected');

    // Extract base sentiment (before the warning)
    const baseSentiment = sentiment.split('⚠️')[0].trim();

    let type = '';
    if (baseSentiment.toLowerCase().includes('positive'))      type = 'positive';
    else if (baseSentiment.toLowerCase().includes('neutral'))  type = 'neutral';
    else if (baseSentiment.toLowerCase().includes('negative')) type = 'negative';

    if (type) setActive(type);

    // Build result HTML
    let resultHTML = `
      <div class="result-statement">"${textInput}"</div>
      <div class="result-label">${baseSentiment}</div>
    `;

    if (isSarcastic) {
      resultHTML += `<div class="sarcasm-warning">⚠️ Sarcasm Detected — result may be inaccurate</div>`;
    }

    resultEl.innerHTML = resultHTML;
    document.getElementById('textInput').value = '';

  } catch (err) {
    resultEl.innerHTML = '<span class="error">Failed to analyze sentiment</span>';
    console.error(err);
  }
}

// Press Enter to submit
document.getElementById('textInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') analyzeSentiment();
});