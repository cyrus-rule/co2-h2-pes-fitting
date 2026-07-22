# YUMI interface contract

The supplied production file is treated as the authority for structure. The
repository does not infer a maximum term list or rewrite an ambiguous header.

## Current mapping

\[
(\ell_1,0,\ell_2,L)_{YUMI}
\longleftrightarrow
(\ell_1,\ell_2,L)_{repository}.
\]

The parser preserves title and header lines verbatim, reads a caller-specified
radial count, and discovers four-index coefficient blocks. Masked coefficient
tokens may be read as structural placeholders, but a final file cannot be
serialized until every value is finite.

When filling a template:

- values are joined by tuple and radius, never by row position;
- a required term omitted by a reduced method is written as `0.0`;
- NaN and infinity are rejected;
- the written file is parsed again in tests;
- terms outside the candidate basis must be investigated before use.

## Open questions requiring group confirmation

1. Does header value `68` count anisotropic terms while excluding `V000`?
2. Is the 69-block template the complete production maximum, or should a larger
   list be supplied with additional zeros?
3. What numeric width and precision are preferred by the production reader?
4. Can an unmasked known-good coefficient file be retained as a golden fixture?

Until those are answered, this module validates structure and round trips but
does not claim production certification.
