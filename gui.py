import PySimpleGUI as sg

class Menu:
    def __init__(self, saved_faces):
        self.saved_faces = saved_faces
        self.saved_name_targeted = False

        self.layout = [
            [sg.Text("Name of Person to Find: "), sg.InputText(key="target_name", enable_events=True)],
            [sg.Text("Insert Images: "), sg.InputText(key="image_paths"), sg.FilesBrowse(key="image_browse")],
            [sg.Text("Saved Faces:")],
            [sg.Listbox(values=self.saved_faces, key="saved_target", size=(30, 10), enable_events=True)],
            [sg.Text("Insert Folder to Sort: "), sg.InputText(key="folder_path"), sg.FolderBrowse()],
            [sg.Button("Submit")],
        ]

        self.window = sg.Window("Simple GUI", self.layout)

    def run(self):
        while True:
            event, values = self.window.read()

            if event == sg.WINDOW_CLOSED:
                break

            elif event == "target_name":
                self.saved_name_targeted = False
                self.window["image_paths"].update(visible=True)
                self.window["image_browse"].update(visible=True)

            elif event == "saved_target":
                self.saved_name_targeted = True
                self.selected_name = values["saved_target"]
                self.window["target_name"].update(self.selected_name)
                self.window["image_paths"].update(visible=False)
                self.window["image_browse"].update(visible=False)

            elif event == "Submit":
                if self.saved_name_targeted:
                    target_name = self.selected_name[0]
                    folder_to_search = values["folder_path"]
                    return target_name, folder_to_search
                else:
                    target_name = values["target_name"]
                    train_images = values["image_paths"].split(';')
                    folder_to_search = values["folder_path"]

                    return target_name, train_images, folder_to_search


        self.window.close()
