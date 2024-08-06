# main.py
import sys

def main(input_string):
    # Convert the input string to uppercase
    uppercase_string = input_string.upper()
    print("Uppercase String:", uppercase_string)

if __name__ == "__main__":
    # Take the input string from command-line arguments
    if len(sys.argv) > 1:
        input_string = sys.argv[1]
    else:
        input_string = ""
    main(input_string)
