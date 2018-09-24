import unittest
from melon_data_parser import parser

# inherits the testcases from unittest in the class TestParser
class TestParser(unittest.TestCase):
    
    def setUp(self):
        print("Test 1: Loading input data")
        try:
            self.dp = parser()
        except:
            self.fail("Data not loaded!")    
        
    def test_get_illegal_intensity(self):
        print("Test 2: Illegal input for MSO intensity")
        try:
            self.dp.get_intensity(intensity=100)
        except:
            self.assertRaises(KeyError)

    def test_get_legal_intensity(self):
        print("Test 3: Normal input for MSO intensity")
        self.dp.get_intensity(intensity=30)   
    
    def test_get_illegal_channel(self):
        print("Test 4: Illegal channel input")
        try:
            self.dp.get_intensity(intensity=30)   
            self.dp.get_channel(channel=100)
        except:
            self.assertRaises(KeyError)
    
    def test_get_legal_channel(self):
        print("Test 4: Normal channel input")
        self.dp.get_intensity(intensity=30)   
        self.dp.get_channel(channel=30)


if __name__ == '__main__':
    unittest.main()
    
