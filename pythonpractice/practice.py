def isprime(number):
    count = 0
    for i in range(2, number):
        if number % i == 0:
            count = count+1
    if count == 0:
        return True
    else:
        return False


user_input = int(input("Enter a number:"))
if isprime(user_input):
    print("The number is a prime number")
else:
    print("The number is not a prime number")
