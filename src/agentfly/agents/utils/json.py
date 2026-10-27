"""Lenient JSON parsing for model output (:func:`jsonish`): repairs invalid escapes and line breaks before decoding."""


def jsonish(text: str) -> dict:
    """
    Convert a JSON-like string (possibly containing invalid escapes and line breaks) to a valid JSON string.
    Process through a state machine: only handle line breaks and invalid escapes within strings.
    """
    valid_escapes = '"\\/bfnrtu'  # JSON valid escape characters
    in_string = False
    escape_active = False
    result_chars = []

    for char in text.strip():
        if not in_string:
            # Direct copy outside string
            result_chars.append(char)
            if char == '"':
                in_string = True  # Enter string area
        else:
            if escape_active:
                # Handle escape sequences
                if char in valid_escapes:
                    result_chars.append(char)
                else:
                    result_chars.append("\\")  # Invalid escape: add extra backslash
                    result_chars.append(char)
                escape_active = False
            else:
                if char == "\\":
                    escape_active = True  # Mark escape start
                    result_chars.append(char)
                elif char == '"':
                    in_string = False  # Exit string area
                    result_chars.append(char)
                else:
                    # Line break in string is converted to escape sequence
                    if char == "\n":
                        result_chars.append("\\n")
                    else:
                        result_chars.append(char)

    fixed_text = "".join(result_chars)
    return fixed_text
