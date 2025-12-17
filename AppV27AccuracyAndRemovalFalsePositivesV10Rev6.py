import cv2
import numpy as np
from flask import Flask, request, render_template, session, redirect, url_for
import logging
from io import BytesIO
import base64
from PIL import Image
import os
import secrets
import uuid  # Import the uuid module
import json  # Import the json module

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)
app.config['SESSION_TYPE'] = 'filesystem'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for default parameter values
DEFAULT_SCALE_FACTOR = 1.05
DEFAULT_MIN_NEIGHBORS = 6
DEFAULT_MIN_SIZE = 30
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# HTML template (Corrected version with escaped curly braces)
HTML_TEMPLATE_START = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Artificial Intelligence Facial Similarity Analyzer</title>
    <style>
        body {{
            font-family: sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}

        #comparison-matrix {{
            display: grid;
            grid-template-columns: auto repeat(auto-fit, minmax(400px, 1fr)); /* Columns: Labels + Surveillance Images */
            gap: 10px;
            max-width: 1500px;
            margin: 20px auto;
        }}
        /* ADDED CSS */
         #comparison-matrix > div:nth-child(odd) {{
             background-color: #f9f9f9; /* light gray */
         }}
        /* ADDED CSS */

        .matrix-cell {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }}

        .face-comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr; /* Three columns: Face 1, Face 2, Heatmap */
            gap: 5px;
            align-items: center;  /* Center items vertically */
        }}

        .face-comparison-grid img {{
            max-width: 100%;
            max-height: 100px;
            display: block;
            margin: 0 auto;
            border: 1px solid #eee;
        }}
         /* ADDED CSS */
        .face-comparison-grid div {{ /* Targets the individual comparison divs */
            border: 1px solid #ddd; /* Subtle border */
            padding: 5px;
            margin-bottom: 5px; /* Add a bit of space between comparisons */
            text-align: center;
        }}
        /* ADDED CSS */

        .matrix-cell img {{
            max-width: 100%;
            max-height: 150px;
            display: block;
            margin: 0 auto;
            border: 1px solid #eee;
        }}

        .matrix-header {{
            font-weight: bold;
            background-color: #f0f0f0;
        }}

        #explanation-box {{
            max-width: 1200px;
            margin: 20px auto;
            border: 1px solid #ccc;
            padding: 15px;
        }}

        .ai-title {{
            font-size: 2em;
            font-weight: bold;
            color: #000;
            background-color: #eee;
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3);
            text-align: center;
        }}

        h2 {{
            text-align: center;
        }}

        #upload-area {{
            max-width: 1200px;
            margin: 20px auto;
            text-align: center;
        }}

        #parameter-area {{
            max-width: 1200px;
            margin: 20px auto;
            text-align: center;
            border: 1px solid #ccc;
            padding: 15px;
        }}
         /* Style for the face detection image container */
        .face-detection-container {{
            position: relative; /* Make the container relative */
            display: inline-block; /* Adjust to content */
            text-align: center; /* Ensure content is centered */
        }}

        /* Style for the remove face button */
        .remove-face-button {{
            position: absolute; /* Position it absolutely within the container */
            background-color: rgba(255, 0, 0, 0.5); /* Semi-transparent red */
            color: white;
            border: none;
            cursor: pointer;
            padding: 2px 5px;
            font-size: 0.8em;
        }}

        /* Highlight on hover */
        .face-detection-container:hover {{
            border: 2px solid blue; /* Highlight the container */
        }}

        /* Visual link on checkbox hover */
        .face-detection-container input[type="checkbox"]:hover + label {{
            background-color: rgba(0, 255, 0, 0.5); /* Slightly darker green on hover */
        }}

        /* Style for the original images with bounding boxes */
        .original-image-container {{
            position: relative;
            display: inline-block;
        }}

        .bounding-box {{
            position: absolute;
            border: 2px solid red;
        }}

    </style>
    <script>
        function removeFace(faceId) {{
            // Call removeSelectedFaces with a single face ID
            removeSelectedFaces([faceId]);
        }}

        function removeSelectedFaces(faceIds) {{
            // If faceIds is not provided, get the selected face IDs from checkboxes
            if (!faceIds) {{
                faceIds = Array.from(document.querySelectorAll('input[name="selected_faces"]:checked'))
                    .map(checkbox => checkbox.value);
            }}

            if (faceIds.length === 0) {{
                alert('No faces selected for removal.');
                return;
            }}

            const formData = new FormData();
            formData.append('selected_face_ids', JSON.stringify(faceIds));
            formData.append('scaleFactor', document.getElementById('scaleFactor').value);
            formData.append('minNeighbors', document.getElementById('minNeighbors').value);
            formData.append('minSize', document.getElementById('minSize').value);
            formData.append('similarityThreshold', document.getElementById('similarityThreshold').value);

            fetch('/remove_faces', {{
                method: 'POST',
                body: formData,
            }})
            .then(response => response.text())
            .then(updatedMatrixHtml => {{
                document.getElementById('comparison-matrix').outerHTML = updatedMatrixHtml; // Replace the comparison-matrix
            }})
            .catch(error => console.error('Error:', error));
        }}

        function generateComparisonMatrix() {{
            const formData = new FormData(document.getElementById('faceSelectionForm')); // Use the form to collect parameters
            const params = new URLSearchParams(new FormData(document.getElementById('uploadForm'))).toString();

            fetch('/generate_matrix?' + params, {{
                method: 'POST',
                 body: new FormData(document.getElementById('faceSelectionForm'))
            }})
            .then(response => response.text())
            .then(comparisonMatrixHtml => {{
                document.getElementById('comparison-matrix').innerHTML = comparisonMatrixHtml; // Render the comparison matrix
            }})
            .catch(error => console.error('Error:', error));
        }}

        function resetAnalysis() {{
             fetch('/reset')
            .then(response => {{
                if (response.redirected) {{
                    window.location.href = response.url; // Redirect to the upload page
                }} else {{
                    console.error('Redirection failed.');
                }}
            }})
            .catch(error => console.error('Error:', error));
        }}

    </script>
</head>
<body>
    <h1 class="ai-title">ARTIFICIAL INTELLIGENCE - HAAR CASCADE CLASSIFIER - CELEBRITY FACIAL RECOGNITION - SEE WHO YOU LOOK LIKE! - LEARN AI IMAGE CLASSIFICATION TECHNIQUES!  </h1>

    <div id="upload-area">
        <h2>Upload Images</h2>
        <form id="uploadForm" method="POST" enctype="multipart/form-data" action="/">
            <label for="referenceImageInput">Reference Images (Rows):</label>
            <input type="file" id="referenceImageInput" name="referenceImage" multiple><br><br>

            <label for="surveillanceImageInput">Surveillance Images (Columns):</label>
            <input type="file" id="surveillanceImageInput" name="surveillanceImage" multiple><br><br>

            <div id="parameter-area">
                <h2>Adjust Parameters</h2>
                <label for="scaleFactor">Scale Factor:</label>
                <input type="number" id="scaleFactor" name="scaleFactor" value="{scale_factor}" step="0.01"><br><br>

                <label for="minNeighbors">Min Neighbors:</label>
                <input type="number" id="minNeighbors" name="minNeighbors" value="{min_neighbors}" step="1"><br><br>

                <label for="minSize">Min Size:</label>
                <input type="number" id="minSize" name="minSize" value="{min_size}" step="1"><br><br>

                <label for="similarityThreshold">Similarity Threshold:</label>
                <input type="number" id="similarityThreshold" name="similarityThreshold" value="{similarity_threshold}" step="0.01"><br><br>
            </div>

            <button type="submit">Upload Images JPG/JPEG, PNG, WebP</button>
        </form>
    </div>

    <div id="explanation-box">
        <h2>How This Works</h2>
        <p>This application compares face images using the following steps:</p>
        <ul>
            <li><strong>Face Detection:</strong> Detects faces in each image using a Haar Cascade Classifier.</li>
            <li>Cropped face regions are converted to grayscale.</li>
            <li>Generates a histogram representing the distribution of grayscale values.</li>
            <li>Calculates a similarity score (correlation) by comparing the histograms.</li>
        </ul>
        <p>A higher score indicates greater similarity.</p>
        <p><strong>AI vs. Traditional:</strong> Uses AI for face detection and traditional image processing for comparison.</p>

        <h2>Parameter Tuning Guide</h2>
        <p>Adjust the following parameters to improve face detection accuracy:</p>
        <ul>
            <li><strong>Scale Factor:</strong>
                <ul>
                    <li>Lower values (e.g., 1.02) are more sensitive but can cause more false positives.</li>
                    <li>Higher values (e.g., 1.06) are faster but might miss smaller faces.</li>
                    <li>*Missing Faces:* Decrease Scale Factor (e.g., 1.04 to 1.03 or 1.02)</li>
                </ul>
            </li>
            <li><strong>Min Neighbors:</strong>
                <ul>
                    <li>Specifies the minimum number of neighboring rectangles that must be detected to consider it a face.</li>
                    <li>Higher values reduce false positives.</li>
                    <li>*Detecting Non-Face Objects:* Increase Min Neighbors (e.g., 6 to 8)</li>
                </ul>
            </li>
            <li><strong>Min Size:</strong>
                <ul>
                    <li>Specifies the minimum size (in pixels) of a face.</li>
                    <li>Increase to eliminate very small false positives.</li>
                </ul>
            </li>
        </ul>

        <h3>Troubleshooting Tips:</h3>
        <ul>
            <li><strong>False Positives (Detecting Non-Face Objects):</strong>
                <ol>
                    <li>Increase <strong>Min Neighbors</strong> first.</li>
                    <li>If that doesn't work, slightly increase <strong>Scale Factor</strong>.</li>
                    <li>If detecting very small objects, increase <strong>Min Size</strong>.</li>
                </ol>
            </li>
            <li><strong>False Negatives (Missing Faces):</strong>
                <ol>
                    <li>Decrease <strong>Scale Factor</strong>. This is the most common solution.</li>
                    <li>If decreasing Scale Factor causes too many false positives, try decreasing Min Neighbors slightly.</li>
                </ol>
            </li>
        </ul>
        <p>Iteratively adjust these parameters and test on a variety of images to find the optimal settings for your specific use case.</p>
    </div>

    <!-- Section for Displaying Images with Bounding Boxes and Checkboxes -->
    <div id="face-selection-area">
        <h2>Select Faces to Remove</h2>
        <form id="faceSelectionForm" method="POST" enctype="multipart/form-data">
        <!-- The images with bounding boxes and checkboxes will be inserted here -->
        {face_selection_html}
        <button type="button" onclick="generateComparisonMatrix()">Generate Comparison Matrix</button>
        </form>
    </div>

    <div id="comparison-matrix">
        <!-- The comparison matrix will be inserted here -->
    </div>

    <button type="button" onclick="resetAnalysis()">Start New Analysis</button>

"""

HTML_TEMPLATE_END = """
</body>
</html>
"""

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def detect_faces(image, scale_factor, min_neighbors, min_size):
    """Detects faces in an image using the Haar cascade classifier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                          minSize=(min_size, min_size))
    return faces


def crop_face(image, face):
    """Crops a face from an image."""
    x, y, w, h = face
    return image[y:y + h, x:x + w]


def calculate_histogram(face_image):
    """Calculates the histogram of a face image."""
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
    hist /= hist.sum()  # Normalize the histogram
    return hist


def compare_histograms(hist1, hist2):
    """Compares two histograms using the correlation method."""
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity


def convert_image_to_base64(image):
    """Converts a OpenCV image to base64 for embedding in HTML."""
    _, buffer = cv2.imencode('.jpg', image)
    io_buf = BytesIO(buffer)
    img_str = base64.b64encode(io_buf.read()).decode('utf-8')
    return img_str

def draw_bounding_boxes(image, faces):
    """Draws bounding boxes on an image with the (x, y) coordinates as the ID."""
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
        # Add the (x, y) coordinates as the ID, positioning it above the box
        cv2.putText(image, f"({x},{y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image


def create_heatmap_image(similarity_score, size=(100, 100), colormap=cv2.COLORMAP_INFERNO):
    """
    Generates a heatmap image based on the similarity score.

    Args:
        similarity_score (float): The similarity score between two faces (0 to 1).
        size (tuple): The desired size (width, height) of the heatmap image.  Defaults to (100, 100).
        colormap (int): The OpenCV colormap to use. Defaults to cv2.COLORMAP_INFERNO.

    Returns:
        str: Base64 encoded string of the heatmap image.
    """

    # Convert the similarity score to a value between 0 and 255
    color_index = int(similarity_score * 255)

    # Create a 1x1 image with the color index
    heatmap_img = np.zeros((1, 1, 1), dtype=np.uint8)
    heatmap_img[0, 0] = color_index

    # Apply the colormap
    heatmap_img = cv2.applyColorMap(heatmap_img, colormap)

    # Resize the heatmap image
    heatmap_img = cv2.resize(heatmap_img, size, interpolation=cv2.INTER_LINEAR)

    # Convert the heatmap image to base64
    heatmap_base64 = convert_image_to_base64(heatmap_img)

    return heatmap_base64



def fix_image_orientation(image_path):  # Changed to accept image path
    """Fixes the orientation of an image loaded from file data."""
    try:
        image = Image.open(image_path)  # Open image from path
        # Check for EXIF data and orientation tag
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif:
                orientation = exif.get(274)
                if orientation:
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(-90, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        # Convert to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        logging.error(f"Error fixing image orientation: {e}")
        return None


def create_face_selection_html(image_paths, scale_factor, min_neighbors, min_size, removed_faces):
    """
    Generates HTML for displaying images with detected faces and checkboxes.
    """
    html = '<h2>Face Selection</h2>'
    for image_path in image_paths:
        image = fix_image_orientation(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            continue

        faces = detect_faces(image, scale_factor, min_neighbors, min_size)
        image_with_boxes = draw_bounding_boxes(image.copy(), faces)
        image_base64 = convert_image_to_base64(image_with_boxes)

        html += f'<div><img src="data:image/jpeg;base64,{image_base64}" alt="Image with Bounding Boxes"/>'

        # Add checkboxes for each detected face
        for x, y, w, h in faces:
            face_id = f'face_{os.path.basename(image_path)}_{x}_{y}_{w}_{h}'
            html += f'<div class="face-detection-container">'
            html += f'<input type="checkbox" id="remove_{face_id}" name="selected_faces" value="{face_id}" {"checked" if face_id in removed_faces else ""}>'
            html += f'<label for="remove_{face_id}">Remove Face ({x},{y})</label>'
            html += '</div>'
        html += '</div>'

    # Add the "Remove Selected Faces" button
    html += '''
        <button type="button" onclick="removeSelectedFaces()">Remove Selected Faces</button>
    '''
    return html

import cv2
import numpy as np
import os
from io import BytesIO
import base64
from PIL import Image
import logging


import cv2
import numpy as np
from io import BytesIO
import base64


def convert_image_to_base64(image):
    """Converts a OpenCV image to base64 for embedding in HTML."""
    _, buffer = cv2.imencode('.jpg', image)
    io_buf = BytesIO(buffer)
    img_str = base64.b64encode(io_buf.read()).decode('utf-8')
    return img_str


def create_custom_colormap(similarity_score, size=(500, 500)):
    """
    Generates a heatmap image with shades of blue,
    where darker blues indicate higher similarity and lighter blues indicate lower similarity.

    Args:
        similarity_score (float): The similarity score between two faces (0 to 1).
        size (tuple): The desired size (width, height) of the heatmap image. Defaults to (100, 100).

    Returns:
        str: Base64 encoded string of the heatmap image.
    """

    # Invert the similarity score so that 0 means "most similar" and 1 means "least similar"
    inverted_score = 1 - similarity_score

    # Convert the inverted score to a value between 0 and 255
    color_index = int(inverted_score * 255)

    # Create a 1x1 image with the color index
    heatmap_img = np.zeros((1, 1, 3), dtype=np.uint8)  # 3 channels for BGR

    # Shades of Blue Colormap
    heatmap_img[0, 0, 0] = 255 - color_index  # Blue (decreases as color_index increases)
    heatmap_img[0, 0, 1] = 255 - color_index  # Green (decreases as color_index increases)
    heatmap_img[0, 0, 2] = 255 - color_index  # Red (decreases as color_index increases)

    # Resize the heatmap image
    heatmap_img = cv2.resize(heatmap_img, size, interpolation=cv2.INTER_LINEAR)

    # Convert the heatmap image to base64
    heatmap_base64 = convert_image_to_base64(heatmap_img)

    return heatmap_base64


def create_greyscale_difference_heatmap(face_image1, face_image2, target_size=(64, 64)):
    """
    Generates a heatmap image showing the pixel-wise difference between two greyscale face images.

    Args:
        face_image1 (numpy.ndarray): The first greyscale face image.
        face_image2 (numpy.ndarray): The second greyscale face image.
        target_size (tuple): The size to which both images will be resized before comparison.

    Returns:
        str: Base64 encoded string of the heatmap image.
    """
    try:
        # Resize images to the same dimensions
        resized_face1 = cv2.resize(face_image1, target_size)
        resized_face2 = cv2.resize(face_image2, target_size)

        # Calculate the absolute difference
        difference = cv2.absdiff(resized_face1, resized_face2)

        # Normalize the difference to the range 0-255
        normalized_difference = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Apply a colormap
        heatmap = cv2.applyColorMap(normalized_difference, cv2.COLORMAP_INFERNO)

        # Convert the heatmap to base64
        heatmap_base64 = convert_image_to_base64(heatmap)
        return heatmap_base64

    except Exception as e:
        print(f"Error creating greyscale difference heatmap: {e}")
        return None


def create_comparison_matrix(reference_image_paths, surveillance_image_paths, scale_factor, min_neighbors, min_size,
                             similarity_threshold, removed_faces):
    """
    Generates the HTML for the face comparison matrix, excluding removed faces.
    """
    reference_faces_data = []
    surveillance_faces_data = []
    html = '<div id="comparison-matrix">'  # Start rendering the comparison matrix

    # Define the target size for face crops
    target_size = (128, 128)  # Example: 128x128 pixels

    # Load reference images and detect faces
    for image_path in reference_image_paths:
        image = fix_image_orientation(image_path)
        if image is None:
            logging.error(f"Failed to load reference image: {image_path}")
            continue

        # Detect faces in reference image
        faces = detect_faces(image, scale_factor, min_neighbors, min_size)
        filtered_faces = [face for face in faces if
                          f'face_{os.path.basename(image_path)}_{face[0]}_{face[1]}_{face[2]}_{face[3]}' not in removed_faces]
        reference_faces_data.append({"filename": os.path.basename(image_path), "faces": filtered_faces, "image": image})

    # Load surveillance images and detect faces
    for image_path in surveillance_image_paths:
        image = fix_image_orientation(image_path)
        if image is None:
            logging.error(f"Failed to load surveillance image: {image_path}")
            continue

        # Detect faces in surveillance image
        faces = detect_faces(image, scale_factor, min_neighbors, min_size)
        filtered_faces = [face for face in faces if
                          f'face_{os.path.basename(image_path)}_{face[0]}_{face[1]}_{face[2]}_{face[3]}' not in removed_faces]
        surveillance_faces_data.append(
            {"filename": os.path.basename(image_path), "faces": filtered_faces, "image": image})

    # Generate the comparison grid
    for ref_data in reference_faces_data:
        for ref_face in ref_data["faces"]:
            ref_face_id = f'face_{ref_data["filename"]}_{ref_face[0]}_{ref_face[1]}_{ref_face[2]}_{ref_face[3]}'

            # Skip this face if it's in the removed list
            if ref_face_id in removed_faces:
                continue

            ref_crop = crop_face(ref_data["image"], ref_face)
            # Resize the reference face crop
            ref_crop_resized = cv2.resize(ref_crop, target_size)
            ref_gray = cv2.cvtColor(ref_crop_resized, cv2.COLOR_BGR2GRAY)  # Get grayscale image
            ref_hist = calculate_histogram(ref_crop_resized)

            for sur_data in surveillance_faces_data:
                for sur_face in sur_data["faces"]:
                    sur_face_id = f'face_{sur_data["filename"]}_{sur_face[0]}_{sur_face[1]}_{sur_face[2]}_{sur_face[3]}'

                    # Skip this face if it's in the removed list
                    if sur_face_id in removed_faces:
                        continue

                    sur_crop = crop_face(sur_data["image"], sur_face)
                    # Resize the surveillance face crop
                    sur_crop_resized = cv2.resize(sur_crop, target_size)
                    sur_gray = cv2.cvtColor(sur_crop_resized, cv2.COLOR_BGR2GRAY)  # Get grayscale image
                    sur_hist = calculate_histogram(sur_crop_resized)

                    # Calculate similarity
                    similarity_score = compare_histograms(ref_hist, sur_hist)
                    heatmap_base64 = create_heatmap_image(similarity_score)
                    # Create greyscale difference heatmap
                    difference_heatmap_base64 = create_greyscale_difference_heatmap(ref_gray, sur_gray)

                    # Add the result to HTML
                    html += f'<div class="matrix-cell">'
                    html += f'<img src="data:image/jpeg;base64,{convert_image_to_base64(ref_crop_resized)}" alt="Reference Face"/>'
                    html += f'<img src="data:image/jpeg;base64,{convert_image_to_base64(sur_crop_resized)}" alt="Surveillance Face"/>'
                    html += f'<img src="data:image/jpeg;base64,{heatmap_base64}" alt="Similarity Heatmap"/>'
                    if difference_heatmap_base64:
                        html += f'<img src="data:image/jpeg;base64,{difference_heatmap_base64}" alt="Greyscale Difference Heatmap"/>'
                    html += f'<p>Similarity: {similarity_score:.2f}</p>'
                    html += '</div>'

    html += '</div>'
    return html



@app.route("/", methods=['GET', 'POST'])
def upload_images():
    # Reset removed_faces when new images are uploaded
    session['removed_faces'] = []
    session.modified = True

    scale_factor = DEFAULT_SCALE_FACTOR
    min_neighbors = DEFAULT_MIN_NEIGHBORS
    min_size = DEFAULT_MIN_SIZE
    similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD

    reference_files = []
    surveillance_files = []

    if request.method == 'POST':
        reference_files = request.files.getlist('referenceImage')
        surveillance_files = request.files.getlist('surveillanceImage')

        # Get parameter values from the form
        try:
            scale_factor = float(request.form.get('scaleFactor', DEFAULT_SCALE_FACTOR))
            min_neighbors = int(request.form.get('minNeighbors', DEFAULT_MIN_NEIGHBORS))
            min_size = int(request.form.get('minSize', DEFAULT_MIN_SIZE))
            similarity_threshold = float(request.form.get('similarityThreshold', DEFAULT_SIMILARITY_THRESHOLD))
        except ValueError:
            logging.error("Invalid parameter value received from form.")
            # Handle the error, maybe set to default values or show an error message

        # Save files to the server and store paths in the session
        reference_image_paths = []
        for file in reference_files:
            # Generate a unique filename
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            reference_image_paths.append(filepath)

        surveillance_image_paths = []
        for file in surveillance_files:
            # Generate a unique filename
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            surveillance_image_paths.append(filepath)

        session['reference_image_paths'] = reference_image_paths
        session['surveillance_image_paths'] = surveillance_image_paths

        session['scale_factor'] = scale_factor
        session['min_neighbors'] = min_neighbors
        session['min_size'] = min_size
        session['similarity_threshold'] = similarity_threshold

        session.modified = True
    else:
        # Clear image paths on GET request
        session.pop('reference_image_paths', None)
        session.pop('surveillance_image_paths', None)
        session.modified = True

    # Retrieve image paths from session if available
    reference_image_paths = session.get('reference_image_paths', [])
    surveillance_image_paths = session.get('surveillance_image_paths', [])

    scale_factor = session.get('scale_factor', DEFAULT_SCALE_FACTOR)
    min_neighbors = session.get('min_neighbors', DEFAULT_MIN_NEIGHBORS)
    min_size = session.get('min_size', DEFAULT_MIN_SIZE)
    similarity_threshold = session.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)

    removed_faces = session.get('removed_faces', [])

    # Generate face selection HTML
    all_image_paths = reference_image_paths + surveillance_image_paths
    face_selection_html = create_face_selection_html(all_image_paths, scale_factor, min_neighbors, min_size, removed_faces)

    comparison_matrix_html = ""

    # Render the template with the face selection
    final_html = HTML_TEMPLATE_START.format(scale_factor=scale_factor, min_neighbors=min_neighbors,
                                            min_size=min_size,
                                            similarity_threshold=similarity_threshold,
                                            face_selection_html=face_selection_html) + comparison_matrix_html + HTML_TEMPLATE_END
    return final_html


@app.route('/generate_matrix', methods=['POST'])
def generate_matrix():
    """
    Generates the comparison matrix after the user selects faces to remove.
    """
    scale_factor = float(request.args.get('scaleFactor'))
    min_neighbors = int(request.args.get('minNeighbors'))
    min_size = int(request.args.get('minSize'))
    similarity_threshold = float(request.args.get('similarityThreshold'))

    reference_image_paths = session.get('reference_image_paths', [])
    surveillance_image_paths = session.get('surveillance_image_paths', [])
    removed_faces = session.get('removed_faces', [])

    comparison_matrix_html = create_comparison_matrix(reference_image_paths, surveillance_image_paths, scale_factor,
                                                      min_neighbors, min_size, similarity_threshold, removed_faces)
    return comparison_matrix_html


@app.route('/remove_faces', methods=['POST'])
def remove_faces():
    """
    Removes selected faces and regenerates the comparison matrix.
    """
    selected_face_ids_json = request.form.get('selected_face_ids')
    selected_face_ids = json.loads(selected_face_ids_json) if selected_face_ids_json else []

    scale_factor = float(request.form.get('scaleFactor'))
    min_neighbors = int(request.form.get('minNeighbors'))
    min_size = int(request.form.get('minSize'))
    similarity_threshold = float(request.form.get('similarityThreshold'))

    # Retrieve the current list of removed faces from the session
    removed_faces = session.get('removed_faces', [])

    # Add the newly selected faces to the list
    for face_id in selected_face_ids:
        if face_id not in removed_faces:
            removed_faces.append(face_id)

    # Store the updated list of removed faces in the session
    session['removed_faces'] = removed_faces
    session.modified = True

    # Retrieve image paths from session
    reference_image_paths = session.get('reference_image_paths', [])
    surveillance_image_paths = session.get('surveillance_image_paths', [])

    # Regenerate face selection HTML with updated removed faces
    all_image_paths = reference_image_paths + surveillance_image_paths
    face_selection_html = create_face_selection_html(all_image_paths, scale_factor, min_neighbors, min_size,
                                                     removed_faces)

    # Regenerate the comparison matrix with the removed faces
    comparison_matrix_html = create_comparison_matrix(reference_image_paths, surveillance_image_paths, scale_factor,
                                                      min_neighbors, min_size, similarity_threshold, removed_faces)

    # Return the updated HTML content for the comparison matrix and face selection area
    return comparison_matrix_html



@app.route('/reset')
def reset():
    """Resets the session and redirects to the upload page."""
    session.clear()  # Clear all session data
    return redirect(url_for('upload_images'))  # Redirect to the upload page


if __name__ == '__main__':
    app.run(debug=True)