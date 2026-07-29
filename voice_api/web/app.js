const state = {
  voices: [],
  lastTtsUrl: null,
  lastCloneUrl: null,
};

const QUALITY_PRESETS = {
  stable: {
    top_k: 8,
    top_p: 0.82,
    temperature: 0.68,
    repetition_penalty: 1.2,
    speed_factor: 1.0,
    best_of: 4,
  },
  natural: {
    top_k: 12,
    top_p: 0.88,
    temperature: 0.78,
    repetition_penalty: 1.16,
    speed_factor: 1.05,
    best_of: 4,
  },
  expressive: {
    top_k: 16,
    top_p: 0.92,
    temperature: 0.86,
    repetition_penalty: 1.12,
    speed_factor: 1.06,
    best_of: 6,
  },
  emotional: {
    top_k: 18,
    top_p: 0.94,
    temperature: 0.92,
    repetition_penalty: 1.08,
    speed_factor: 1.04,
    best_of: 6,
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function setMessage(id, text, type = "") {
  const element = $(id);
  element.textContent = text;
  element.classList.remove("ok", "error");
  if (type) element.classList.add(type);
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || data.message || JSON.stringify(data);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

function setBusy(form, busy) {
  form.querySelectorAll("button, input, select, textarea").forEach((element) => {
    element.disabled = busy;
  });
}

function switchView(viewName) {
  $$(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.view === viewName);
  });
  $$(".view").forEach((view) => {
    view.classList.toggle("is-active", view.id === `view-${viewName}`);
  });
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function voiceOptions() {
  if (!state.voices.length) {
    return '<option value="">등록된 목소리 없음</option>';
  }
  return state.voices
    .map((voice) => `<option value="${voice.voice_id}">${voice.voice_id}</option>`)
    .join("");
}

function renderVoices() {
  $("#voice-count").textContent = String(state.voices.length);
  $("#tts-voice").innerHTML = voiceOptions();

  const list = $("#voice-list");
  if (!state.voices.length) {
    list.innerHTML = '<p class="message">목소리가 없습니다.</p>';
    return;
  }

  list.innerHTML = state.voices
    .map(
      (voice) => `
        <article class="voice-item">
          <div>
            <div class="voice-title">${voice.voice_id}</div>
            <div class="voice-meta">${voice.prompt_lang} · ${formatDate(voice.created_at)}</div>
          </div>
          <button class="delete" data-delete="${voice.voice_id}" type="button">삭제</button>
        </article>
      `,
    )
    .join("");
}

async function loadHealth() {
  try {
    const health = await requestJson("/health");
    $("#runtime").textContent = `${health.version} · ${health.device} · voices ${health.voices.length}`;
  } catch (error) {
    $("#runtime").textContent = `연결 실패: ${error.message}`;
  }
}

async function loadVoices() {
  const data = await requestJson("/voices");
  state.voices = data.voices || [];
  renderVoices();
  await loadHealth();
}

function formValue(form, name) {
  const element = form.elements[name];
  return element ? element.value.trim() : "";
}

function numberValue(form, name, fallback) {
  const raw = formValue(form, name).replace(",", ".");
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

function decimalFormValue(form, name, fallback) {
  return String(numberValue(form, name, fallback));
}

function presetFor(form) {
  return QUALITY_PRESETS[formValue(form, "quality_preset")] || QUALITY_PRESETS.natural;
}

function applyPreset(form) {
  const preset = presetFor(form);
  Object.entries(preset).forEach(([name, value]) => {
    if (form.elements[name]) form.elements[name].value = value;
  });
}

function ttsPayload(form) {
  return {
    voice_id: formValue(form, "voice_id"),
    text: formValue(form, "text"),
    text_lang: formValue(form, "text_lang") || "ko",
    speed_factor: numberValue(form, "speed_factor", 1.05),
    seed: numberValue(form, "seed", -1),
    top_k: numberValue(form, "top_k", 12),
    top_p: numberValue(form, "top_p", 0.88),
    temperature: numberValue(form, "temperature", 0.78),
    repetition_penalty: numberValue(form, "repetition_penalty", 1.16),
    parallel_infer: false,
    best_of: Number(formValue(form, "best_of") || 4),
    retry_bad_segments: true,
  };
}

function setAudio(kind, blob) {
  const urlKey = kind === "clone" ? "lastCloneUrl" : "lastTtsUrl";
  if (state[urlKey]) URL.revokeObjectURL(state[urlKey]);
  state[urlKey] = URL.createObjectURL(blob);

  const player = $(`#${kind}-player`);
  const download = $(`#${kind}-download`);
  player.src = state[urlKey];
  download.href = state[urlKey];
  download.classList.remove("is-hidden");
}

async function generateTts(save = false) {
  const form = $("#tts-form");
  if (!form.reportValidity()) return;
  setBusy(form, true);
  setMessage("#tts-message", save ? "저장 생성 중" : "생성 중");

  try {
    if (save) {
      const payload = {
        ...ttsPayload(form),
        filename: formValue(form, "filename") || undefined,
      };
      const result = await requestJson("/tts/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const audioResponse = await fetch(result.url);
      if (!audioResponse.ok) throw new Error(await readError(audioResponse));
      const blob = await audioResponse.blob();
      setAudio("tts", blob);
      $("#tts-download").download = result.filename;
      setMessage("#tts-message", `저장됨: ${result.path}`, "ok");
      return;
    }

    const response = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ttsPayload(form)),
    });
    if (!response.ok) throw new Error(await readError(response));
    const blob = await response.blob();
    setAudio("tts", blob);
    $("#tts-download").download = "tts.wav";
    setMessage("#tts-message", "생성 완료", "ok");
  } catch (error) {
    setMessage("#tts-message", error.message, "error");
  } finally {
    setBusy(form, false);
  }
}

async function registerVoice(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  data.set("consent_confirmed", form.elements.consent_confirmed.checked ? "true" : "false");
  data.set("overwrite", form.elements.overwrite.checked ? "true" : "false");

  setBusy(form, true);
  setMessage("#voice-message", "등록 중");
  try {
    const response = await fetch("/voices", { method: "POST", body: data });
    if (!response.ok) throw new Error(await readError(response));
    const voice = await response.json();
    setMessage("#voice-message", `${voice.voice_id} 등록 완료`, "ok");
    form.reset();
    await loadVoices();
  } catch (error) {
    setMessage("#voice-message", error.message, "error");
  } finally {
    setBusy(form, false);
  }
}

async function deleteVoice(voiceId) {
  const response = await fetch(`/voices/${encodeURIComponent(voiceId)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await readError(response));
  await loadVoices();
}

async function generateClone(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  data.set("consent_confirmed", form.elements.consent_confirmed.checked ? "true" : "false");
  const preset = presetFor(form);
  data.set("top_k", preset.top_k);
  data.set("top_p", preset.top_p);
  data.set("temperature", preset.temperature);
  data.set("speed_factor", decimalFormValue(form, "speed_factor", preset.speed_factor));
  data.set("best_of", String(numberValue(form, "best_of", preset.best_of)));
  data.set("retry_bad_segments", "true");

  setBusy(form, true);
  setMessage("#clone-message", "생성 중");

  try {
    const response = await fetch("/clone-tts", { method: "POST", body: data });
    if (!response.ok) throw new Error(await readError(response));
    const blob = await response.blob();
    setAudio("clone", blob);
    setMessage("#clone-message", "생성 완료", "ok");
  } catch (error) {
    setMessage("#clone-message", error.message, "error");
  } finally {
    setBusy(form, false);
  }
}

function bindEvents() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });

  $("#voice-form").addEventListener("submit", registerVoice);
  $("#tts-form").addEventListener("submit", (event) => {
    event.preventDefault();
    generateTts(false);
  });
  $("#save-tts").addEventListener("click", () => generateTts(true));
  $("#clone-form").addEventListener("submit", generateClone);
  $("#refresh-voices").addEventListener("click", () => loadVoices());
  $$('select[name="quality_preset"]').forEach((select) => {
    select.addEventListener("change", () => applyPreset(select.form));
    applyPreset(select.form);
  });

  $("#voice-list").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-delete]");
    if (!button) return;
    if (!window.confirm(`${button.dataset.delete} 목소리를 삭제할까요?`)) return;
    button.disabled = true;
    try {
      await deleteVoice(button.dataset.delete);
    } catch (error) {
      setMessage("#voice-message", error.message, "error");
      button.disabled = false;
    }
  });
}

bindEvents();
loadVoices().catch((error) => {
  setMessage("#voice-message", error.message, "error");
});
