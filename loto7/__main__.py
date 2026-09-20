import os
import sys

from .cli import main

if __name__ == "__main__":
    try:
        code = main()
    except BrokenPipeError:
        # `| head` などで出力が途中で閉じられた場合。stderr への追い出しを防ぐ。
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 0
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)
