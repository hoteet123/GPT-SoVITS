# GPT-SoVITS Voice API

이 폴더에는 공식 GPT-SoVITS 코드 위에 얹은 프로그램용 TTS API가 들어 있습니다.
기본 GPT-SoVITS `/tts`보다 쓰기 쉽게, 한 번 등록한 참조 음성을 `voice_id`로 계속 호출할 수 있게 만들었습니다.

## 1. 설치

Python 의존성이 이미 설치되어 있고 모델 파일만 없으면 아래만 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\download_voice_models.ps1 -Source HF
```

완전 새 환경이면 처음 한 번 아래를 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_voice_api.ps1 -Device CPU -Source HF
```

NVIDIA CUDA 환경이면 `CPU` 대신 `CU126` 또는 `CU128`을 사용하세요.

## 2. API 실행

```powershell
powershell -ExecutionPolicy Bypass -File .\run_voice_api.ps1
```

기본 주소는 `http://127.0.0.1:9881` 입니다.
브라우저에서 `http://127.0.0.1:9881/`를 열면 목소리 등록과 TTS 샘플 생성을 할 수 있습니다.
`http://127.0.0.1:9881/docs`에서는 Swagger UI로 API를 직접 테스트할 수 있습니다.

## 3. 목소리 등록

참조 음성은 보통 5-10초 정도의 깨끗한 단일 화자 음성이 좋습니다.
`prompt_text`에는 그 음성 파일에서 실제로 말한 문장을 적어주세요.

```powershell
curl.exe -X POST "http://127.0.0.1:9881/voices" `
  -F "voice_id=my_voice" `
  -F "prompt_text=안녕하세요. 오늘은 날씨가 참 좋네요." `
  -F "prompt_lang=ko" `
  -F "consent_confirmed=true" `
  -F "reference_audio=@C:\path\to\reference.wav"
```

등록된 음성은 `voice_api_data/voices` 아래에 저장됩니다.

## 4. TTS 생성

WAV 파일로 바로 받기:

```powershell
Invoke-WebRequest `
  -Method Post `
  -Uri "http://127.0.0.1:9881/tts" `
  -ContentType "application/json" `
  -Body '{"voice_id":"my_voice","text":"안녕하세요. 프로그램에서 사용할 음성입니다.","text_lang":"ko"}' `
  -OutFile "output.wav"
```

서버에 파일로 저장하고 경로를 JSON으로 받기:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9881/tts/save" `
  -ContentType "application/json" `
  -Body '{"voice_id":"my_voice","text":"저장되는 음성입니다.","text_lang":"ko","filename":"hello.wav"}'
```

## 5. 내 프로그램에서 호출

JavaScript:

```javascript
const response = await fetch("http://127.0.0.1:9881/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    voice_id: "my_voice",
    text: "안녕하세요. 앱에서 재생할 음성입니다.",
    text_lang: "ko"
  })
});

const audioBlob = await response.blob();
const audioUrl = URL.createObjectURL(audioBlob);
new Audio(audioUrl).play();
```

Python 예제는 `examples/python_client.py`에 넣어두었습니다.

## 6. 즉석 참조 음성으로 만들기

음성을 저장하지 않고 한 번만 합성할 수도 있습니다.

```powershell
curl.exe -X POST "http://127.0.0.1:9881/clone-tts" `
  -F "text=이 문장을 참조 음성의 톤으로 읽습니다." `
  -F "text_lang=ko" `
  -F "prompt_text=안녕하세요. 오늘은 날씨가 참 좋네요." `
  -F "prompt_lang=ko" `
  -F "consent_confirmed=true" `
  -F "reference_audio=@C:\path\to\reference.wav" `
  --output clone.wav
```

## 주요 엔드포인트

- `GET /health`: 모델, 언어, 등록된 음성 확인
- `GET /voices`: 등록된 음성 목록
- `POST /voices`: 참조 음성 등록
- `DELETE /voices/{voice_id}`: 등록 음성 삭제
- `POST /tts`: WAV 생성
- `POST /tts/save`: WAV 생성 후 서버에 저장
- `POST /clone-tts`: 업로드한 참조 음성으로 즉석 생성
- `POST /weights`: fine-tuned GPT/SoVITS 가중치로 교체

본인 또는 사용 허락을 받은 목소리만 등록해서 사용하세요.
