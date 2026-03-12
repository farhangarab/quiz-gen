print("this github")
print("Test from brookdale")

#create a function that change celsius to fahrenheit
def celsius_to_fahrenheit(celsius_value: float) -> float:
    return (celsius_value * 9/5) + 32





def password_strength(password: str) -> str:
    score = 0

    if len(password) >= 8:
        score += 1
    if any(char.islower() for char in password):
        score += 1
    if any(char.isupper() for char in password):
        score += 1
    if any(char.isdigit() for char in password):
        score += 1
    if any(not char.isalnum() for char in password):
        score += 1

    if score <= 2:
        return "Weak"
    if score <= 4:
        return "Medium"
    return "Strong"
