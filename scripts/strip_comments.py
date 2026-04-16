import re
from pathlib import Path


def clean_file(source: str) -> str:
    lines = source.split('\n')
    result = []
    in_multiline = False
    ml_char = None

    for raw_line in lines:
        stripped = raw_line.strip()

        if in_multiline:
            if ml_char in raw_line:
                idx = raw_line.find(ml_char)
                after = raw_line[idx + 3:]
                if ml_char not in after:
                    in_multiline = False
            continue

        if stripped.startswith('"""') or stripped.startswith("'''"):
            ml_char = stripped[:3]
            end_count = stripped.count(ml_char)
            if end_count >= 2:
                continue
            in_multiline = True
            continue

        if stripped.startswith('#'):
            continue

        new_chars = []
        in_str = False
        str_char = None
        i = 0
        while i < len(raw_line):
            ch = raw_line[i]
            if in_str:
                new_chars.append(ch)
                if ch == '\\':
                    i += 1
                    if i < len(raw_line):
                        new_chars.append(raw_line[i])
                elif ch == str_char:
                    in_str = False
            elif ch in ('"', "'"):
                triple = raw_line[i:i+3]
                if triple in ('"""', "'''"):
                    ml_char = triple
                    end_idx = raw_line.find(ml_char, i + 3)
                    if end_idx != -1:
                        i = end_idx + 3
                        continue
                    else:
                        in_multiline = True
                        break
                else:
                    in_str = True
                    str_char = ch
                    new_chars.append(ch)
            elif ch == '#':
                break
            else:
                new_chars.append(ch)
            i += 1

        result.append(''.join(new_chars).rstrip())

    text = '\n'.join(result)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n'


if __name__ == '__main__':
    patterns = [
        'src/**/*.py',
        'services/*.py',
        'train/*.py',
        'inference/*.py',
        'scripts/*.py',
    ]
    base = Path('.')
    files = []
    for pat in patterns:
        files.extend(base.glob(pat))

    changed = 0
    for f in sorted(set(files)):
        if f.name == 'strip_comments.py':
            continue
        src = f.read_text(encoding='utf-8')
        clean = clean_file(src)
        f.write_text(clean, encoding='utf-8')
        changed += 1
        print(f'  cleaned: {f}')
    print(f'\nDone. {changed} files cleaned.')
