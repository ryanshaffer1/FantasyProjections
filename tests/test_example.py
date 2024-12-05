import unittest

class TestExample(unittest.TestCase):

    def test_one_equals_one(self):
        self.assertEqual(1,1)

    def test_one_not_equals_two(self):
        self.assertNotEqual(1,2)
    

if __name__ == '__main__':
    unittest.main()