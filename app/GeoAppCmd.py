'''
User input address and proximity through command line
Map is opened in a separate window
'''

import folium
from folium.plugins import MarkerCluster
import math
from geopy.geocoders import Nominatim
from geopy import distance
import pandas as pd
import numpy as np
import spacy
import webbrowser
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView


class GeoAppCmd():
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.locations_file = os.path.join(self.current_dir, "data", "locations.pkl")
        self.model_path = os.path.join(self.current_dir, "..", "address-segmentation", "ner-sg", "output", "models",
                                       "model-best")
        self.location_data = None
        self.nlp = None

    def address_to_lat_long(self, address):
        geolocator = Nominatim(user_agent="myGeocoder")
        location = geolocator.geocode(address)
        if location:
            print(f"\n[geocode] Postal code: {address}, coordinates: ({location.latitude}, {location.longitude})")
            return location
        else:
            # can introduce fallback in the future
            print(f"\nAddress: {address}, Latitude and Longitude not found")
            return None

    def create_folium_map(self, df, input_address, latitude, longitude, proximity_threshold):
        custom_zoom = self.get_zoom_level(proximity_threshold)
        m = folium.Map(location=[latitude, longitude], zoom_start=custom_zoom)

        address_groups = df.groupby(['latitude', 'longitude'])['address'].apply(list).reset_index()
        for _, group in address_groups.iterrows():
            lat, lng = group['latitude'], group['longitude']
            addresses = group['address']
            popup_content = "<br>".join(addresses)
            folium.Marker(location=[lat, lng],
                        popup=folium.Popup(popup_content, max_width=250),
                        icon=folium.Icon(icon='fa-location-dot', color='blue')).add_to(m)

        folium.Marker(location=[latitude, longitude],
                    popup=folium.Popup(input_address, max_width=250),
                    icon=folium.Icon(icon='fa-location-dot', color='red')).add_to(m)

        radius_meters = proximity_threshold * 1000  # Convert proximity threshold from km to meters
        folium.Circle(location=[latitude, longitude],
                    radius=radius_meters,
                    color='red',
                    fill_color='orange',
                    fill_opacity=0.2).add_to(m)

        return m



    def save_folium_map(self, m, map_filepath):
        m.save(map_filepath)

    def display_pyqt_map(self, map_filepath):
        app = QApplication([])

        view = QWebEngineView()
        view.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        view.setWindowState(Qt.WindowMaximized)  # Set the window state to maximized
        view.load(QUrl.fromLocalFile(map_filepath))
        view.show()

        # Resize the window to the maximum available size
        desktop = QApplication.desktop()
        rect = desktop.availableGeometry(view)
        view.setGeometry(rect)

        print("Map generated!\n")

        app.exec_()

    def display_map(self, df, input_address, latitude, longitude, proximity_threshold):
        print("Generating map...\n")
        map_filepath = os.path.join(self.current_dir, "map.html")

        m = self.create_folium_map(df, input_address, latitude, longitude, proximity_threshold)
        self.save_folium_map(m, map_filepath)
        self.display_pyqt_map(map_filepath)

    def km_to_pixels(self, length_km, latitude, zoom):
        earth_radius = 6371
        tile_size = 256
        lat_rad = math.radians(latitude)
        circ = 2 * math.pi * earth_radius * math.cos(lat_rad)
        pixel_length = circ / tile_size
        length_pixels = (length_km / pixel_length) * (2 ** zoom)
        return length_pixels

    def get_zoom_level(self, proximity_threshold):
        min_threshold = 0.1  # Minimum proximity threshold
        max_threshold = 10  # Maximum proximity threshold
        min_zoom = 19  # Minimum zoom level
        max_zoom = 13  # Maximum zoom level

        # Calculate the proportional zoom level
        zoom_level = np.interp(proximity_threshold, [min_threshold, max_threshold], [min_zoom, max_zoom])

        return int(zoom_level)

    def get_user_input(self):
        while True:
            try:
                input_address = input("\nEnter the address: ")
                proximity_threshold = float(input("\nEnter the proximity threshold (in km): "))
                return input_address, proximity_threshold
            except ValueError:
                print("Invalid input. Please try again.")

    def filter_locations(self, location_data, user_latitude, user_longitude, proximity_threshold):
        filtered_locations = location_data[
            location_data.apply(
                lambda row: distance.distance(
                    (user_latitude, user_longitude),
                    (row['latitude'], row['longitude'])
                ).km <= proximity_threshold,
                axis=1
            )
        ]
        return filtered_locations

    def print_addresses(self, df, proximity_threshold):
        addresses = df['address'].tolist()
        count = len(addresses)
        print(f"Number of locations found within {proximity_threshold} km proximity: {count}\n")

        print("-------------------------FILTERED LOCATIONS-------------------------")
        for address in addresses:
            print(address)
        print("--------------------------------------------------------------------\n")

    def load_location_data(self):
        self.location_data = pd.read_pickle(self.locations_file)[['address', 'postal_code', 'latitude', 'longitude']]

    def load_spacy_model(self):
        self.nlp = spacy.load(self.model_path)

    def run(self):
        self.load_location_data()
        self.load_spacy_model()

        while True:
            input_address, proximity_threshold = self.get_user_input()

            doc = self.nlp(input_address)
            ent_list = [(ent.text, ent.label_) for ent in doc.ents]
            postal_code = None
            for ent_text, ent_label in ent_list:
                if ent_label == "POSTAL_CODE":
                    postal_code = ent_text
                    break

            user_location = self.address_to_lat_long(postal_code)
            if user_location is not None:
                user_latitude, user_longitude = user_location.latitude, user_location.longitude
                filtered_locations = self.filter_locations(self.location_data, user_latitude, user_longitude,
                                                           proximity_threshold)

                if not filtered_locations.empty:
                    self.print_addresses(filtered_locations, proximity_threshold)
                    self.display_map(filtered_locations, input_address, user_latitude, user_longitude,
                                     proximity_threshold)
                else:
                    print("No locations found within the specified proximity.")

                break  # Exit the loop after successful inputs and processing
            else:
                print("Unable to retrieve coordinates for the address")


if __name__ == "__main__":
    app = GeoAppCmd()
    app.run()