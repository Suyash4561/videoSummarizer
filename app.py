import os
from flask import Flask, request, jsonify

from google import genai
from google.genai.types import Part, FileData
import google.genai.errors as genai_errors

app = Flask(__name__)

# ------------------------------
# Gemini client setup
# ------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# OR: client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        prompt = data.get("prompt", "")

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return jsonify({"response": response.text})

    except genai_errors.GenAiException as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return "Gemini API is working!"

if __name__ == '__main__':
    app.run(debug=True)
