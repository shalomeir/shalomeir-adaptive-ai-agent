from __future__ import annotations


def escape_newlines_in_string_literals(code: str) -> str:
    """Escape raw newlines inside single-line Python string literals.

    Weak models sometimes put ``\n`` into a JSON string intending an escaped
    newline, which becomes an actual newline inside a Python quote after JSON
    decoding. Python then raises ``SyntaxError: unterminated string literal``.
    Escaping only newlines while inside non-triple-quoted strings preserves code
    line structure and keeps multiline triple-quoted strings intact.
    """
    out: list[str] = []
    quote: str | None = None
    triple = False
    escaped = False
    i = 0
    while i < len(code):
        char = code[i]
        if quote is not None:
            if escaped:
                out.append(char)
                escaped = False
            elif char == "\\":
                out.append(char)
                escaped = True
            elif char == quote:
                if triple and code[i : i + 3] == quote * 3:
                    out.append(quote * 3)
                    i += 2
                    quote = None
                    triple = False
                else:
                    out.append(char)
                    if not triple:
                        quote = None
            elif char == "\n" and not triple:
                out.append("\\n")
            else:
                out.append(char)
            i += 1
            continue

        if char in {"'", '"'}:
            quote = char
            if code[i : i + 3] == char * 3:
                triple = True
                out.append(char * 3)
                i += 3
                continue
            triple = False
        out.append(char)
        i += 1
    return "".join(out)
