# -*- coding: utf-8 -*-
#!/usr/bin/env python
from cairlib.DialogueStatistics import DialogueStatistics
from cairlib.DialogueState import DialogueState
from cairlib.DialogueTurn import DialogueTurn
from cairlib.CAIRclient_alterego_utils import Utils
from playsound import playsound
from gtts import gTTS
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
from mutagen.mp3 import MP3
from dotenv import load_dotenv, find_dotenv
from cairclient_alterego_vision.srv import GestureService
import xml.etree.ElementTree as ET
import pygame.camera
import threading
import requests
import argparse
import socket
import base64
import random
import zlib
import json
import time
import xml
import sys
import re
import os
import io
import openai
import rospy
import rospkg

rp = rospkg.RosPack()
package_path = rp.get_path('cairclient_alterego_vision')
folder_path = package_path + "/common" 

# Location of the server
cineca = "131.175.205.146"
local = "130.251.13.130"
server_ip = cineca
audio_recorder_ip = "130.251.13.173"
registration_ip = local
language = "it"
max_history_turns = 6

openai.organization = "org-OWePijhLCGVSJWhT7TQXBK7D"
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# OpenAI
openai = True
if openai:
    server_port = "5005"
else:
    server_port = "5000"
request_uri = "http://" + server_ip + ":" + server_port + "/CAIR_hub"

# Dense captioning - if set to false the dense_cap_result will be sent empty to the server and the
# visual information will not be used by gpt-4
dense_cap = True
dense_cap_result = []
img_port = "5010"
img_url = "http://" + server_ip + ":" + img_port + "/CAIR_dense_captioning"


def stream_and_play(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text,
    )
    # Convert the binary response content to a byte stream
    byte_stream = io.BytesIO(response.content)
    # Read the audio data from the byte stream
    audio = AudioSegment.from_file(byte_stream, format="mp3")
    # Play the audio
    play(audio)


def mp3_duration(path):
    try:
        audio = MP3(path)
        length = audio.info.length
        return length
    except:
        return None
        
class CAIRclient:
    def __init__(self):
        # Instances of the classes of the other files in the libs folder containing functions needed here
        self.utils = Utils(language, server_port, server_ip, registration_ip)
        self.isAlive = True
        self.exit_keywords = ["stop talking", "esci dallapp", "esci dallapplicazione", "quit the application"]
        self.repeat_keywords = ["repeat", "can you repeat", "say it again", "puoi ripetere", "ripeti", "non ho capito"]
        self.dialogue_state_file_path = os.path.join(folder_path, "dialogue_state.json")
        self.speakers_info_file_path = os.path.join(folder_path, "speakers_info.json")
        self.dialogue_statistics_file_path = os.path.join(folder_path, "dialogue_statistics.json")
        self.nuance_vectors_file_path = os.path.join(folder_path, "nuance_vectors.json")
        self.camera_image_path = os.path.join(folder_path, "camera_image.jpg")
        self.audio_file_path = os.path.join(folder_path, "audio.mp3")
	
        # To store the previous sentence said by the robot
        self.previous_sentence = ""
        self.dialogue_sentence = []
        self.dialogue_state = {}
        self.dialogue_statistics = {}
        self.speakers_info = {}
        self.nuance_vectors = {}
        self.conversation_history = {}
        self.plan_sentence = ""
        self.plan = ""
        # This variable tells if the user want the robot to repeat a sentence
        self.repeat = False
        
        # Read the sentences in the correct language from the file
        sentences_file_path = os.path.join(folder_path, "sentences_" + language + ".txt")
        self.sentences = []
        with open(sentences_file_path) as f:
            self.sentences = [line.rstrip() for line in f]

        self.dense_cap = dense_cap
        self.dense_cap_result = dense_cap_result
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.offset = 0
        
    def gesture_service_client(self, filename, audio_duration, offset):
        # Blocking call that waits for server
        rospy.wait_for_service('gesture_service')
        try:
            gesture_service = rospy.ServiceProxy('gesture_service', GestureService)
            response = gesture_service(filename, audio_duration, offset)
            self.offset = response.offset
            rospy.loginfo("**NEW OFFSET: %f", self.offset)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
        
    def print_used_nuances(self, dialogue_nuances):
        print("________USED NUANCES________")
        for key, value in dialogue_nuances["flags"].items():
            flag_index = value.index(1)
            if flag_index == len(value) - 1:
                print(key + ":", dialogue_nuances["values"][key])
            else:
                print(key + ":", dialogue_nuances["values"][key][flag_index])
        print("____________________________")

    def acquire_image(self, cam):
        print("Image thread: start acquiring camera input")
        cam.start()
        # grab first frame
        img = cam.get_image()
        WIDTH = img.get_width()
        HEIGHT = img.get_height()
        # display = pygame.display.set_mode((WIDTH, HEIGHT), 0)
        screen = pygame.surface.Surface((WIDTH, HEIGHT))

        while True:
            # Capture an image
            screen = cam.get_image(screen)
            # Save the image
            pygame.image.save(screen, self.camera_image_path)
            with open(self.camera_image_path, "rb") as img_file:
                # Encode your image with base64 and convert it to string
                img_encoded = base64.b64encode(img_file.read()).decode('utf-8')
            # Create a dictionary with the encoded image
            data = {"frame": img_encoded}
            # Send the image to the server using a POST request
            response = requests.post(img_url, json=data)
            # Print the server's response
            self.dense_cap_result = response.json()["result"]
            time.sleep(3.5)

    def hub_request(self, data):
        encoded_data = json.dumps(data).encode('utf-8')
        compressed_data = zlib.compress(encoded_data)
        hub_response = requests.put(request_uri, data=compressed_data, verify=False)
        # If the Hub cannot contact the dialogue service, the response will be empty
        if hub_response:
            # Overwrite the array containing the states of the profiles with those contained in the Hub response
            # The speakers info are not sent to the Hub.
            self.dialogue_state = DialogueState(d=hub_response.json()['dialogue_state'])
            # Store the updated dialogue state in the file
            with open(self.dialogue_state_file_path, 'w') as f:
                json.dump(self.dialogue_state.to_dict(), f, ensure_ascii=False, indent=4)
            self.dialogue_sentence = hub_response.json()['dialogue_sentence']

            if data["req_type"] == 1:
                # self.print_used_nuances(self.dialogue_state.dialogue_nuances)
                self.dialogue_statistics = DialogueStatistics(d=hub_response.json()["dialogue_statistics"])
                # The hub updates the average topic distance matrix, hence it should be written on the file
                with open(self.dialogue_statistics_file_path, 'w') as f:
                    json.dump(self.dialogue_statistics.to_dict(), f, ensure_ascii=False, indent=4)
                self.plan_sentence = hub_response.json()['plan_sentence']
                self.plan = hub_response.json()['plan']
        else:
            print("No response received from the Hub!")
            exit(0)

    def start_dialogue(self):
        # Try connecting to the socket that records the audio
        print("Trying to connect to the audio recorder socket.")
        try:
            self.client_socket.connect((audio_recorder_ip, 9090))
        except ConnectionError:
            if language == "it":
                to_say = "Mi dispiace, non riesco a connettermi al microfono. Controlla l'indirizzo I P e riprova."
            else:
                to_say = "I'm sorry, I can't connect to the microphone. Check the IP address and try again."
            print("R:", to_say)
            tts = gTTS(to_say, lang=language)
            tts.save(self.audio_file_path)
            playsound(self.audio_file_path)
            sys.exit(1)

        # If it's the first time using the system, call the function that acquires the first state
        if not os.path.isfile(self.speakers_info_file_path):
            print("First user!")
            # This function creates the speakers_info and the speakers_sequence_stats files and initializes them
            # with the info of a generic user
            welcome_sentence_str = self.utils.acquire_initial_state()
            # Retrieve starting nuance vectors
            with open(self.nuance_vectors_file_path) as f:
                self.nuance_vectors = json.load(f)
            welcome_str = welcome_sentence_str
        else:
            print("Users are already present in the info file")
            if language == "it":
                welcome_back_msg = "È bello rivedervi! Di cosa vorreste parlare?"
            else:
                welcome_back_msg = "Welcome back! I missed you. What would you like to talk about?"
            welcome_str = welcome_back_msg

        print("R:", welcome_str)
        tts = gTTS(welcome_str, lang=language)
        tts.save(self.audio_file_path)
        duration = mp3_duration(self.audio_file_path)
        filename = "talk" + str(random.randint(0, 9)) + ".bag"
        welcome_thread = threading.Thread(None, self.gesture_service_client, args=(filename, duration, 0,))
        welcome_thread.start()
        time.sleep(1)
        playsound(self.audio_file_path)
        # stream_and_play(welcome_str)
        self.previous_sentence = welcome_str

        # Retrieve the state of the conversation and save it in a dictionary
        with open(self.dialogue_state_file_path) as f:
            self.dialogue_state = DialogueState(d=json.load(f))
        # If it is the first time, fill the nuance vectors from the file
        if len(self.nuance_vectors) != 0:
            self.dialogue_state.dialogue_nuances = self.nuance_vectors
        # Store the welcome or welcome back string in the assistant field
        self.dialogue_state.conversation_history.append({"role": "assistant", "content": welcome_str})

        # Retrieve the info of the users and store them in a dictionary
        with open(self.speakers_info_file_path) as f:
            self.speakers_info = json.load(f)

        # Retrieve dialogue statistics file
        with open(self.dialogue_statistics_file_path) as f:
            self.dialogue_statistics = DialogueStatistics(d=json.load(f))

        self.dialogue_state.prev_dialogue_sentence = [["s", self.previous_sentence]]
        prev_turn_last_speaker = ""
        prev_speaker_topic = ""

        # If dense captioning should be used, start the thread to update visual information
        if self.dense_cap:
            # Initialize camera
            pygame.camera.init()
            cam_list = pygame.camera.list_cameras()
            cam = pygame.camera.Camera(cam_list[0])
            t1 = threading.Thread(None, self.acquire_image, args=(cam,))
            t1.start()

        while self.isAlive:
            self.offset = 0.0
            filename = "talk" + str(random.randint(0, 9)) + ".bag"
            print("** Listening **")
            if os.path.exists(self.audio_file_path):
                os.remove(self.audio_file_path)
            # Tell the audio recorder that the client is ready to receive the user reply
            self.client_socket.send(self.dialogue_state.sentence_type.encode("utf-8"))
            xml_string = self.client_socket.recv(1024).decode('utf-8')

            if xml_string == "":
                if language == "it":
                    to_say = "Mi dispiace, c'è stato qualche problema con la connessione al microfono esterno."
                else:
                    to_say = "I'm sorry, there was a problem with the connection to the external microphone."
                print("R:", to_say)
                tts = gTTS(to_say, lang=language)
                tts.save(self.audio_file_path)
                # duration = mp3_duration(self.audio_file_path)
                # mic_err_thread = threading.Thread(None, gesture_service_client, args=("talk", duration, self.offset,))
                # mic_err_thread.start()
                playsound(self.audio_file_path)
                # stream_and_play(to_say)
                sys.exit(1)

            # Do not proceed until the xml string is complete and all tags are closed
            proceed = False
            while not proceed:
                try:
                    ET.ElementTree(ET.fromstring(xml_string))
                    proceed = True
                except xml.etree.ElementTree.ParseError:
                    # If the xml is not complete, read again from the socket
                    print("The XML is not complete.")
                    xml_string = xml_string + self.client_socket.recv(1024).decode('utf-8')

            # Create a dialogue turn object starting from the xml
            dialogue_turn = DialogueTurn(xml_string)

            # Update the dialogue statistics only if the required minimum number of users is registered
            if len(self.dialogue_statistics.mapping_index_speaker) > 1:
                self.dialogue_statistics.update_statistics(dialogue_turn, prev_turn_last_speaker)

                # Update content of the speaker stats file after having updated them after someone talked
                with open(self.dialogue_statistics_file_path, 'w') as cl_state:
                    json.dump(self.dialogue_statistics.to_dict(), cl_state, ensure_ascii=False, indent=4)

            # Parse the xml string and extract the first sentence and the first speaker
            tree = ET.ElementTree(ET.fromstring(xml_string))
            speaker_id = tree.findall('profile_id')[0]
            sentence = tree.findall('profile_id')[0].text.strip('.,!?')
            print("U:", sentence)

            # Check if the user wants to exit or wants the robot to repeat the previous sentence
            sentence = sentence.replace(".", "")
            # Reset repeat to false, otherwise it will always repeat the previous sentence
            self.repeat = False

            # If the user said one of the "Exit Application keywords"
            if any(exit_sent in sentence for exit_sent in self.exit_keywords):
                self.isAlive = False
                if language == "it":
                    goodbye_msg = "Ok, è stato bello passare del tempo insieme! A presto!"
                else:
                    goodbye_msg = "Ok, it was a pleasure talking with you! Goodbye."
                print("R:", goodbye_msg)
                tts = gTTS(goodbye_msg, lang=language)
                tts.save(self.audio_file_path)
                duration = mp3_duration(self.audio_file_path)
                goodbye_thread = threading.Thread(None, self.gesture_service_client, args=(filename, duration, self.offset,))
                goodbye_thread.start()
                time.sleep(1)
                playsound(self.audio_file_path)
                sys.exit(0)
            # If the user said a Repeat keyword
            elif sentence.lower() in self.repeat_keywords:
                # If a previous sentence to repeat exists
                if self.previous_sentence:
                    self.repeat = True
                    if language == "it":
                        repeat_msg = "Certamente. Ho detto: "
                    else:
                        repeat_msg = "Sure! I said: "
                    print("R: " + repeat_msg + self.previous_sentence)
                    tts = gTTS(repeat_msg + self.previous_sentence, lang=language)
                    tts.save(self.audio_file_path)
                    playsound(self.audio_file_path)
                else:
                    if language == "it":
                        repeat_msg = "Mi dispiace, non ho niente da ripetere."
                    else:
                        repeat_msg = "I'm sorry, I have nothing to repeat."
                    print("R:", repeat_msg)
                    tts = gTTS(repeat_msg, lang=language)
                    tts.save(self.audio_file_path)
                    playsound(self.audio_file_path)
                    # os.system("afplay audio.mp3")

            # If the user did not ask to exit or to repeat something, send the sentence to the server
            if not self.repeat:
                # Store the user sentence in the conversation history of the dialogue state and pop the first item if needed
                if len(self.dialogue_state.conversation_history) >= max_history_turns:
                    self.dialogue_state.conversation_history.pop(0)
                self.dialogue_state.conversation_history.append({"role": "user", "content": sentence})

                # Copy the speakers info in a dictionary that does not contain the names
                # This is needed by OpenAI as it should know the gender.
                speakers_info_no_names = {}
                for speaker_id in self.speakers_info:
                    speakers_info_no_names[speaker_id] = {"gender": self.speakers_info[speaker_id]["gender"],
                                                          "age": self.speakers_info[speaker_id]["age"]}

                # Compose the payload of the message to be sent to the server
                data = {"req_type": 1, "client_sentence": xml_string, "dialogue_state": self.dialogue_state.to_dict(),
                        "dialogue_statistics": self.dialogue_statistics.to_dict(),
                        "speakers_info": speakers_info_no_names,
                        "prev_speaker_info": {"id": prev_turn_last_speaker, "topic": prev_speaker_topic},
                        "dense_cap_result": self.dense_cap_result}

                # Update the info about id and topic of previous speaker to the current one
                prev_turn_last_speaker = dialogue_turn.turn_pieces[-1].profile_id
                prev_speaker_topic = self.dialogue_state.topic

                # Create the thread for the first request
                req_thread = threading.Thread(target=self.hub_request, args=(data,))
                req_thread.start()

                # # When using openAI say something to fill the void while waiting
                if openai:
                    random_sent = random.choice(self.sentences)
                    print("R:", random_sent)
                    
                    tts = gTTS(random_sent, lang=language)
                    tts.save(self.audio_file_path)
                    duration = mp3_duration(self.audio_file_path)
                    rand_sent_thread = threading.Thread(None, self.gesture_service_client, args=(filename, duration, self.offset,))
                    rand_sent_thread.start()
                    time.sleep(1)
                    playsound(self.audio_file_path)
                    # stream_and_play(random_sent)

                # Wait for the thread to finish
                req_thread.join()

                # If there is a plan sentence, it means that something has been matched by the Plan manager service
                if self.plan_sentence:
                    print(str(self.plan_sentence))
                    self.utils.replace_speaker_name(self.plan_sentence, self.speakers_info)
                    self.plan_sentence = self.utils.replace_schwa_in_string(self.plan_sentence, self.speakers_info,
                                                                            speaker_id)
                    print("R:", self.plan_sentence)
                    tts = gTTS(self.plan_sentence, lang=language)
                    tts.save(self.audio_file_path)
                    duration = mp3_duration(self.audio_file_path)
                    plan_sent_thread = threading.Thread(None, self.gesture_service_client, args=(filename, duration, self.offset,))
                    plan_sent_thread.start()
                    time.sleep(1)
                    playsound(self.audio_file_path)
                    # os.system("afplay audio.mp3")
                    # stream_and_play(self.plan_sentence)

                # If there is a plan, execute it (if the behavior is installed)
                if self.plan:
                    plan_items = self.plan.split("#")[1:]
                    # For each action in the plan, check which action is it and execute it
                    for item in plan_items:
                        action = re.findall("action=(\w+)", item)[0]
                        if action == "registration":
                            # The function that manages the registration, updates the files and returns the updated
                            # dictionaries, so that we don't have to read from the files at each turn.
                            self.speakers_info, self.dialogue_statistics = self.utils.registration_procedure()
                        else:
                            item = item.encode('utf-8')
                            if language == "it":
                                to_say = "Mi dispiace, non sono ancora in grado di svolgere azioni."
                            else:
                                to_say = "I'm sorry, I am still not able to perform actions."
                            print("R:", to_say)
                            # tts = gTTS(to_say, lang=language)
                            # tts.save("audio.mp3")
                            # playsound("audio.mp3")

                # Make a copy of the dialogue sentence before it is modified by the second request
                dialogue_sentence1 = self.dialogue_sentence
                dialogue_sentence1_history = self.utils.process_sentence(dialogue_sentence1, self.speakers_info)
                # Store the assistant sentence in the conversation history of the dialogue state
                if len(self.dialogue_state.conversation_history) >= max_history_turns:
                    self.dialogue_state.conversation_history.pop(0)
                self.dialogue_state.conversation_history.append(
                    {"role": "assistant", "content": dialogue_sentence1_history})

                dialogue_sentence1_str = self.utils.replace_speaker_name(dialogue_sentence1_history, self.speakers_info)
                if openai:
                    # Update the data content for the second request - update also the dialogue state!
                    data["req_type"] = 2
                    data["dialogue_state"] = self.dialogue_state.to_dict()
                    data["dense_cap_result"] = self.dense_cap_result
                    # Create a thread that performs another request to get the continuation of the dialogue
                    req_thread = threading.Thread(target=self.hub_request, args=(data,))
                    req_thread.start()

                print("R:", dialogue_sentence1_str)
            
                tts = gTTS(dialogue_sentence1_str, lang=language)
                tts.save(self.audio_file_path)
                duration = mp3_duration(self.audio_file_path)
                dialogue1_thread = threading.Thread(None, self.gesture_service_client, args=(filename, duration, self.offset,))
                # Wait for the previous gesture thread to finish
                rand_sent_thread.join()
                dialogue1_thread.start()
                time.sleep(1)
                playsound(self.audio_file_path)
                # os.system("afplay audio.mp3")
                # stream_and_play(dialogue_sentence1_str)

                if openai:
                    if req_thread.is_alive():
                        req_thread.join()
                    dialogue_sentence2 = self.dialogue_sentence[1:]
                    dialogue_sentence2_history = self.utils.process_sentence(dialogue_sentence2, self.speakers_info)

                    dialogue_sentence2_str = self.utils.replace_speaker_name(dialogue_sentence2_history,
                                                                             self.speakers_info)
                    print("R:", dialogue_sentence2_str)
                    
                    tts = gTTS(dialogue_sentence2_str, lang=language)
                    tts.save(self.audio_file_path)
                    duration = mp3_duration(self.audio_file_path)
                    dialogue2_thread = threading.Thread(None, self.gesture_service_client, args=(filename, duration, self.offset,))
                    # Wait for the previous gesture thread to finish
                    dialogue1_thread.join()
                    dialogue2_thread.start()
                    time.sleep(1)
                    playsound(self.audio_file_path)
                    # stream_and_play(dialogue_sentence2_str)
                else:
                    dialogue_sentence2_history = ""

                    # Replace the last assistant reply with the complete one
                self.dialogue_state.conversation_history.pop()
                self.dialogue_state.conversation_history.append(
                    {"role": "assistant", "content": dialogue_sentence1_history + " " + dialogue_sentence2_history})
                self.dialogue_state.prev_dialogue_sentence = self.dialogue_sentence


if __name__ == '__main__':
    # Define the program description
    text = 'This is the client for CAIR.'
    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text)
    # Add long and short argument
    parser.add_argument("--language", "-l", help="set the language of the client to it or en")
    # Read arguments from the command line
    args = parser.parse_args()
    if not args.language:
        print("No language provided. The default English language will be used.")
    else:
        language = args.language
        print("The language has been set to", language)

    rospy.init_node('gesture_service_client', anonymous=False)
    cair_client = CAIRclient()
    cair_client.start_dialogue()
