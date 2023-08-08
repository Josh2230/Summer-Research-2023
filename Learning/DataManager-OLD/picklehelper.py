import pickle

class PickleHelper:
  def __init__(self) -> None:
      pass

  def save_obj(self, path, obj):
    f = open(path, "wb")
    pickle.dump(obj, f)
    f.close()

  def load_obj(self, path):
    f = open(path, "rb")
    g = pickle.load(f)
    f.close()
    return g