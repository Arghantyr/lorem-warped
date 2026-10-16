# Writing conventions

This documentation uses selected rules from ASD-STE100 Simplified Technical English, Issue 9. ASD published Issue 9 in January 2025.[1] The official site offers the issue as a free download.[2]

This documentation does not have an ASD-STE100 compliance review. Do not treat it as a certified STE document.

## Rules used in this documentation

- Use short sentences. Limit procedure sentences to 20 words. Limit description sentences to 25 words.[1]
- Give one instruction in each procedure sentence. Use the command form for instructions.[1]
- Put related information in one paragraph. Use no more than six sentences in a paragraph.[1]
- Use the same technical term for the same item.[1]
- Use American English spelling unless another rule applies.[1]

Code identifiers, CLI options, file names, and mathematical notation are technical terms in this documentation. Keep their spelling unchanged. The standard permits technical nouns and verbs for a subject field, subject to its rules.[1]

## Technical terms

| Term | Meaning in this documentation |
| --- | --- |
| image | An input tensor or a file that contains pixels. |
| tensor | A PyTorch `torch.Tensor` object. |
| CHW | The image dimension order: channels, height, width. |
| center | A `(y, x)` point that sets the position of a droplet. |
| field | An `H × W × 2` tensor of `(dy, dx)` displacement values. |
| intensity | A multiplier for geometric displacement. The unit is pixels. |
| radius | The distance from a droplet center to its outer limit. The unit is pixels. |
| mode | The selected image operation. |
| profile | The formula that sets droplet displacement. |
| lens | The `sine_lens` geometric page warp. It is not the refraction model. |
| refraction model | The `refractive_index` operation for a uniform slab. |
| page size | The physical width and height used to convert millimeters to pixels. |
| CLI | The command-line interface. |

Use `center` in prose. Keep `centres`, `centre_count`, and other Python identifiers unchanged in code. These identifiers are part of the API.

## Scope

The standard has controlled word meanings, word forms, and a dictionary.[1] This documentation does not verify each word against that dictionary. The writing rules above guide the text, but they do not establish full compliance.

## Sources

[1] https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf — ASD-STE100 Simplified Technical English, Issue 9 (2025)
[2] https://www.asd-ste100.org/STE_downloads.html — ASD-STE100 official downloads
