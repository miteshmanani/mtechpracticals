from flask import Flask, request

app = Flask(__name__)


@app.route('/save', methods=['POST'])
def save_blob():
    with open("downloaded_video.mp4", "wb") as f:
        f.write(request.data)
    return "File saved", 200


if __name__ == "__main__":
    app.run(debug=True)

# "blob:https://player.vimeo.com/16a097da-fa65-453d-b3f9-0582a0a58cb5"
blob: https: // player.vimeo.com/b860fee0-bba6-424f-84e7-cbc111b993f6
