from DataManager.picklehelper import PickleHelper


class TestClassInstance:
    x = 100
    def __init__(self):
        self.y = 200

T = TestClassInstance()
PH = PickleHelper()
PH.save_obj('test.b', T)

TLB = PH.load_obj('test.b')
print(TestClassInstance.x)
print(TLB.y)