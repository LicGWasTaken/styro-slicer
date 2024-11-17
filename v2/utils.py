import colorama

def msg(str, *argvs):
    if "error" in argvs:
        print(colorama.Style.BRIGHT + colorama.Fore.RED + str)
    elif "warning" in argvs:
        print(colorama.Style.NORMAL + colorama.Fore.YELLOW + str)
    elif "process" in argvs:
        print(colorama.Style.NORMAL + str + "...")
    elif "info" in argvs:
        print(colorama.Style.DIM + "--> " + str)
    elif "debug" in argvs:
        print(colorama.Style.BRIGHT + colorama.Fore.LIGHTBLUE_EX + str)

    print(colorama.Style.RESET_ALL, end="")

def is_structured(list_: list, format: list):
    """Recursively compare the structure (dimensions and var types) of two lists"""

    # Variable length
    if len(format) < 2:
        for var in list_:
            if not is_structured(var, format[0]):
                return 0
        return 1

    # Fixed length
    else:
        if len(format) != len(list_):
            return 0

        for i, var in enumerate(list_):
            if not isinstance(var, list):
                if not isinstance(var, format[i]):
                    return 0
            elif not is_structured(var, format[i]):
                return 0
        return 1

