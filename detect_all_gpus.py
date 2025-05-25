import wmi


def list_all_gpus():
    w = wmi.WMI()
    gpus = w.Win32_VideoController()
    for gpu in gpus:
        print(f"Name: {gpu.Name}")
        print(f"Driver Version: {gpu.DriverVersion}")
        print(f"Video Processor: {gpu.VideoProcessor}")
        print(f"Adapter RAM: {int(gpu.AdapterRAM) / 1024 / 1024} MB")
        print('-' * 40)


list_all_gpus()
