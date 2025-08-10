# generate_keys.py
import secrets

def generate_key(length: int = 32) -> str:
    """
    Generates a cryptographically secure, URL-safe string.
    
    Args:
        length (int): The desired length of the key in bytes. 
                      32 bytes will result in a 43-character URL-safe string.
                      For hex, 32 bytes results in a 64-character string.

    Returns:
        str: A hex-encoded secure random string.
    """
    return secrets.token_hex(length)

if __name__ == "__main__":
    # Generate a key for Flask's SECRET_KEY
    flask_secret_key = generate_key()
    
    # Generate a key for JWT_SECRET_KEY
    jwt_secret_key = generate_key()

    print("Generated secure keys for your .env file:\n")
    print("="*50)
    print(f"SECRET_KEY='{flask_secret_key}'")
    print(f"JWT_SECRET_KEY='{jwt_secret_key}'")
    print("="*50)
    print("\nCopy these lines and paste them into your .env file.")

