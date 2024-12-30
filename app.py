'''from flask import Flask, render_template, jsonify, request
from fullcode import process_dataset  # Assuming processor.py is in the same directory

app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process():
    try:
        # Hardcode or define the file path dynamically
        file_path = 'Fashion_Retail_Sales.csv'  # Replace with your dataset's file path
        
        # Process the dataset and get the output
        output = process_dataset(file_path)

        # For POST: If the output needs to be returned as JSON
       

        # For GET: Render the output on a webpage
        return render_template('output.html', output=output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
'''
from flask import Flask, request, jsonify, render_template
from fullcode import process_dataset  # Import your dataset processing logic

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    try:
        # Retrieve the file path from the JSON payload
        data = request.json  # Parse JSON payload
        file_path = data.get('file_path')  # Retrieve file_path key

        if not file_path:
            return jsonify({"error": "File path not provided"}), 400

        # Process the dataset using the provided file path
        output = process_dataset(file_path)
        return jsonify(output)  # Return the processed output as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
