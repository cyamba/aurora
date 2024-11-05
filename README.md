# Aurora Project

Aurora is a Speech-to-Text (STT), Language Model (LLM), and Text-to-Speech (TTS) integration project. The project aims to create a natural, conversational AI experience by utilizing voice input, processing it through a language model, and returning spoken responses.

## Features

- **STT**: Converts spoken input to text.
- **LLM**: Processes the text input using an advanced language model to generate responses.
- **TTS**: Converts the generated text response back into speech output.

## Requirements

- Python 3.8 or higher
- Required libraries: `torch`, `pygame`, `sounddevice`, `gTTS`, `vosk`, `transformers`

## Model Download

To use the Speech-to-Text functionality, you need to download the Vosk model. Follow these steps:

1. Go to [Vosk Models Download](https://alphacephei.com/vosk/models).
2. Download the appropriate model for your needs (e.g., `vosk-model-small-en-us-0.15.zip`).
3. Extract the downloaded model and place it in the `models/` directory in your project root.

Your project directory should look like this:

```
aurora/
├── aurora/               # Main source code directory
│   ├── main.py           # Main script to run the project
├── models/               # Directory for storing downloaded models
│   └── vosk-model-small-en-us-0.15/  # Place the extracted model here
├── README.md             # Project description and usage information
├── requirements.txt      # Dependencies
└── LICENSE               # License information
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/aurora.git
   cd aurora
   ```
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Download the Vosk model and place it in the `models/` directory as specified above.

## Usage

Run the main script to initiate the conversation process. Speak into the microphone, and the system will respond with an appropriate answer.

```sh
python aurora/main.py
```

## License

This project is licensed under a permissive license inspired by MIT, which encourages creativity and innovation. You are free to use, clone, modify, and distribute this code for both non-commercial and commercial purposes, provided that appropriate credit is given to the original author. Any derivative works must maintain the same open and permissive spirit, ensuring that they can also be freely used, modified, and distributed by others.

