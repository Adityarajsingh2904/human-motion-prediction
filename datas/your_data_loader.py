import logging

import numpy as np

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = None

    def load(self):
        logging.info("Loading data from %s", self.path)
        try:
            self.data = np.loadtxt(self.path, delimiter=',')
        except Exception as e:
            logging.error("Failed to load data: %s", e)
            raise
        logging.info("Finished loading data")
        return self.data

    def preprocess(self):
        logging.info("Preprocessing data")
        if self.data is None:
            raise ValueError("No data loaded")
        self.data = (self.data - np.mean(self.data, axis=0)) / (np.std(self.data, axis=0) + 1e-8)
        logging.info("Finished preprocessing data")
        return self.data

    def save(self, out_path):
        logging.info("Saving data to %s", out_path)
        if self.data is None:
            raise ValueError("No data to save")
        np.savetxt(out_path, self.data, delimiter=',')
        logging.info("Finished saving data")
