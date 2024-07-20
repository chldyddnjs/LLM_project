from unittest import TestCase
from models.llama3.tokenizer import Tokenizer,ChatFormat


# TOKENIZER_PATH=<path> python -m unittest llama/test_tokenizer.py

class TokenizerTests(TestCase):
    def __init__(self):
        self.tokenizer = Tokenizer('o200k_base')
        self.format = ChatFormat(self.tokenizer)

    def test_special_tokens(self):
        self.assertEqual(
            self.tokenizer.special_tokens["<|begin_of_text|>"],
            199998,
        )

    def test_encode(self):
        self.assertEqual(
            self.tokenizer.encode(
                "This is a test sentence.",
                bos=True,
                eos=True
            ),
            [128000, 2028, 374, 264, 1296, 11914, 13, 128001],
        )
    
    def test_encode_batch(self):
        enc_b = self.tokenizer.encode_batch(
            ["This is a test sentence1.","This is a test sentence2."],
            bos=True,
            eos=True,
        )

        print([self.tokenizer.decode(enc) for enc in enc_b])

    def test_decode(self):
        self.assertEqual(
            self.tokenizer.decode(
                [128000, 2028, 374, 264, 1296, 11914, 13, 128001],
            ),
            "<|begin_of_text|>This is a test sentence.<|end_of_text|>",
        )

    def test_encode_message(self):
        message = {
            'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May\n\nHow many clips did Natalia sell in May?', 
            'role': 'user'
        }
        print(self.format.encode_message(message))
        print(self.tokenizer.decode(self.format.encode_message(message)))
        # self.assertEqual(
        #     self.format.encode_message(message),
        #     [
        #         128006,  # <|start_header_id|>
        #         882,  # "user"
        #         128007,  # <|end_header_id|>
        #         271,  # "\n\n"
        #         2028, 374, 264, 1296, 11914, 13,  # This is a test sentence.
        #         128009,  # <|eot_id|>
        #     ]
        # )

    def test_encode_dialog(self):
        dialog = [
            {
                'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May\n\nHow many clips did Natalia sell in May?', 
                'role': 'user'
            }, 
            {
                'content': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.', 
                'role': 'assistant'
            }, 
            {
                'content': 'How many clips did Natalia sell altogether in April and May?', 
                'role': 'user'
            }, 
            {
                'content': 'Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.', 
                'role': 'assistant'
            }
        ]
        print(self.tokenizer.decode(self.format.encode_dialog_prompt(dialog)))
        # self.assertEqual(
        #     self.format.encode_dialog_prompt(dialog),
        #     [
        #         128000,  # <|begin_of_text|>
        #         128006,  # <|start_header_id|>
        #         9125,     # "system"
        #         128007,  # <|end_header_id|>
        #         271,     # "\n\n"
        #         2028, 374, 264, 1296, 11914, 13,  # "This is a test sentence."
        #         128009,  # <|eot_id|>
        #         128006,  # <|start_header_id|>
        #         882,     # "user"
        #         128007,  # <|end_header_id|>
        #         271,     # "\n\n"
        #         2028, 374, 264, 2077, 13,  # "This is a response.",
        #         128009,  # <|eot_id|>
        #         128006,  # <|start_header_id|>
        #         78191,   # "assistant"
        #         128007,  # <|end_header_id|>
        #         271,     # "\n\n"
        #     ]
        # )
if __name__=="__main__":
    tok = TokenizerTests()
    tok.test_encode_batch()