from flask import Flask
import subprocess

# this code is a flask app for Aaron to hit and call the calibrate function

app = Flask(__name__)

@app.route('/calibrate')
def call_calib():
    try:
        # Run the calib.py script as a separate process
        process = subprocess.Popen(['python', '/Users/davidhe/Documents/GitHub/hackthe6ix2025/last/calib.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Return the output of the calib.py script
        if process.returncode == 0:
            return f"Calibration completed successfully:\n{stdout.decode()}"
        else:
            return f"Calibration failed:\n{stderr.decode()}"
    except Exception as e:
        return f"An error occurred while running calibration: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)