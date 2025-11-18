import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------------------
# Gemini import & client setup
# ------------------------------
try:
    from google import genai
    from google.genai.types import Part, FileData
    import google.genai.errors as genai_errors
except ImportError:
    genai = None
    Part = None
    FileData = None
    genai_errors = None

# Initialize client only if library is present and GEMINI_API_KEY exists.
client = None
if genai is not None:
    try:
        # genai.Client() reads GEMINI_API_KEY from environment by default
        client = genai.Client()
    except Exception as e:
        # If key missing or invalid, client remains None and endpoints will return 500.
        print(f"Warning: could not initialize Gemini client: {e}")
        client = None
else:
    print("Warning: google-genai library not installed. Install with pip install google-genai.")

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ------------------------------
# Simple in-memory DB for users
# ------------------------------
memory_db = {
    "users": {}  # username -> {"password": "<hashed>"}
}

# ------------------------------
# Utilities
# ------------------------------
YOUTUBE_RE = re.compile(
    r'^(https?://)?(www\.)?((youtube\.com/)|(youtu\.be/)).+', re.IGNORECASE
)


def is_valid_youtube_url(url: str) -> bool:
    return bool(url and YOUTUBE_RE.match(url))

# ------------------------------
# SERVE STATIC FILES
# ------------------------------


@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('.', 'index.html')

# ------------------------------
# AUTH ROUTES
# ------------------------------


@app.route('/api/register', methods=['POST'])
def register():
    """
    Register endpoint.
    Expects JSON: {"username": "<str>", "password": "<str>"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"status": "error", "message": "username and password are required"}), 400

    if username in memory_db["users"]:
        return jsonify({"status": "error", "message": "User already exists"}), 409

    memory_db["users"][username] = {
        "password": generate_password_hash(password)}
    return jsonify({"status": "success", "message": "User registered"}), 201


@app.route('/api/login', methods=['POST'])
def login():
    """
    Login endpoint.
    Expects JSON: {"username": "<str>", "password": "<str>"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"status": "error", "message": "username and password are required"}), 400

    user = memory_db["users"].get(username)
    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404

    if not check_password_hash(user["password"], password):
        return jsonify({"status": "error", "message": "Incorrect password"}), 401

    # NOTE: This simple implementation returns success only.
    # For production, return a JWT or session cookie.
    return jsonify({"status": "success", "message": "Login successful"}), 200


# ------------------------------
# YOUTUBE SUMMARIZER ROUTE
# ------------------------------
@app.route('/api/summarize', methods=['POST'])
def summarize():
    """
    Summarize a YouTube video using Gemini model "gemini-2.5-flash".
    Request JSON:
    {
        "url": "<youtube_url>",
        "prompt": "<optional prompt string>"
    }
    """

    # 1) Ensure the gemini client is initialized
    if client is None:
        return jsonify({
            "status": "error",
            "message": "Gemini client not initialized. Check GEMINI_API_KEY and that google-genai is installed."
        }), 500

    # 2) Parse input
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid or missing JSON body"}), 400

    youtube_url = data.get("url", "").strip()
    prompt_text = data.get(
        "prompt", "Summarize the content of this YouTube video in three bullet points.").strip()

    if not youtube_url:
        return jsonify({"status": "error", "message": "Missing 'url' field"}), 400

    if not is_valid_youtube_url(youtube_url):
        return jsonify({"status": "error", "message": "Invalid YouTube URL"}), 400

    # 3) Build request parts (video + prompt)
    try:
        contents = [
            Part(
                file_data=FileData(
                    file_uri=youtube_url,
                    mime_type="video/mp4"
                )
            ),
            Part.from_text(text=prompt_text)
        ]
    except Exception as e:
        # Defensive: if Part/FileData constructors are missing or incorrect
        return jsonify({"status": "error", "message": f"Internal error preparing request: {e}"}), 500

    # 4) Call Gemini inside try/except and handle quota errors
    try:
        # The call happens only when this endpoint is invoked.
        # Model: gemini-2.5-flash per your request.
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents
        )

        # Attempt to return response.text which most genai responses expose.
        # Fallback: stringify the whole response.
        summary_text = getattr(response, "text", None) or str(response)

        return jsonify({
            "status": "success",
            "summary": summary_text,
            "url": youtube_url
        }), 200

    except Exception as e:
        # Provide helpful error messages for quota (429) and general errors.
        err_str = str(e)
        status_code = 500
        user_msg = f"Gemini API error: {err_str}"

        # If google.genai.errors is available, try to detect ClientError / 429
        try:
            if genai_errors and isinstance(e, genai_errors.ClientError):
                # genai ClientError often contains response_json with 'error' and 'status'
                resp = getattr(e, "response", None)
                # Detect 429 roughly by message content if no numeric attribute
                if "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str or "429" in err_str:
                    status_code = 429
                    user_msg = ("Quota exceeded for Gemini API. "
                                "Free-tier/video processing can use many tokens. "
                                "Consider enabling billing, switching to a cheaper model, or using transcripts instead. "
                                f"Raw error: {err_str}")
        except Exception:
            # ignore detection failure; fall back to generic
            pass

        return jsonify({"status": "error", "message": user_msg}), status_code


# ------------------------------
# Root health check
# ------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"message": "SummarizeTube API (gemini-2.5-flash) - healthy"}), 200


# ------------------------------
# Run server
# ------------------------------
if __name__ == '__main__':
    # NOTE: debug False in production; True is convenient for development.
    app.run(host='0.0.0.0', port=5000, debug=True)