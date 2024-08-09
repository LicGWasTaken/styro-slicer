import colorama

def print_bp(text):
    print('--> ' + text)

def print_error(text):
    print(colorama.Fore.RED + text + colorama.Style.RESET_ALL)