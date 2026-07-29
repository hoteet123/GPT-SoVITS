import json
import urllib.request


API_URL = "http://127.0.0.1:9881"


def synthesize_to_file(text: str, voice_id: str, output_path: str = "output.wav") -> None:
    payload = {
        "voice_id": voice_id,
        "text": text,
        "text_lang": "ko",
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_URL}/tts",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        audio = response.read()

    with open(output_path, "wb") as file:
        file.write(audio)


if __name__ == "__main__":
    synthesize_to_file("안녕하세요. GPT-SoVITS API 테스트입니다.", "my_voice")
