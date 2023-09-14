# grounding-of-formulae

This source code is a hard copy of the current working projects of [babyygemperor/MioGatto](https://github.com/babyygemperor/MioGatto), a fork of [MioGatto](https://github.com/babyygemperor/MioGatto) in support of [my bachelor's thesis](https://github.com/babyygemperor/bachelors-thesis). The results of this thesis are availabe on [Google Sheets](https://docs.google.com/spreadsheets/d/1v0t9q5V2j4phjZxXQFIH3b06vl8UBNrFDZlz6ajPAac/edit?usp=sharing).

There is a docker image added for help.

### How to use

```sh
docker run -p 4100:4100 -d ghcr.io/babyygemperor/miogatto:latest
```

The rendered page is at https://localhost:4100/
This launches it with the default paper in view, i.e. 2107.10832 annotated by gpt-4.

To launch with a paper of your choice, please run

```sh
docker run -p 4100:4100 -d ghcr.io/babyygemperor/miogatto:latest python -m server <PAPER_ID>
```

You can find the paper ID from the folder of ./sources; do not include the extension in the PAPER_ID.

#### Example usage

```sh
docker run -p 4100:4100 -d ghcr.io/babyygemperor/miogatto:latest python -m server 1906.05149-StableBeluga2
```

In order to run the tool to generate GPT generated annotations, use the following commands.

```sh
$ export OPENAI_API_KEY=...
$ python -m extra.annotate.annotator path/to/paper.html
```

The expected output of the annotation looks like this. (Note: Annotation takes time, please be patient for output, it can take several minutes)
```txt
extra.annotate.annotator INFO: Begin to annotate Paper "1906.05149"
extra.annotate.annotator INFO: Started at 2023-09-14 21:51:09.237771
extra.annotate.annotator INFO: Using gpt-3.5-turbo
extra.annotate.annotator INFO: Starting dictionary generation
extra.annotate.annotator INFO: Completed dictionary generation
extra.annotate.annotator INFO: Time taken: 0:00:38.784364
extra.annotate.annotator INFO: 20691 tokens @ $0.0015-0.002/1K Tokens = $0.032619 in 40 seconds => 517.275 Tokens/second
extra.annotate.annotator INFO: Starting annotation
extra.annotate.annotator INFO: 0% Completed
extra.annotate.annotator INFO: 10% Completed
extra.annotate.annotator INFO: 20% Completed
extra.annotate.annotator INFO: 30% Completed
extra.annotate.annotator INFO: 40% Completed
extra.annotate.annotator INFO: 50% Completed
extra.annotate.annotator INFO: 60% Completed
extra.annotate.annotator INFO: 70% Completed
extra.annotate.annotator INFO: 80% Completed
extra.annotate.annotator INFO: 90% Completed
extra.annotate.annotator INFO: 100% Completed
extra.annotate.annotator INFO: Annotation completed
extra.annotate.annotator INFO: Time taken: 0:00:59.712894
extra.annotate.annotator INFO: 43850 tokens @ $0.0015-0.002/1K Tokens = $0.065848 in 59 seconds => 743.2203389830509 Tokens/second
extra.annotate.annotator INFO: Total concepts annotated: 162 of 162 (100.0%)
extra.annotate.annotator INFO: Total cost: $0.1
extra.annotate.annotator INFO: Total time taken: 0:01:43.027364
```

Note: anno.json and mcdict.json of the paper should exist in ./data folder. If not, please use the following command to generate those files from the html.

```sh
python -m tools.preprocess path/to/file.html -d data
```

In order to generate the html from tex, use the command from line 139 of this README file. (Preparing data with `latexmlc`) 

Note: The annotation tool `extra.annotate.annotator-open` is experimental. It has never been tested. To see the functionings, please see `./notebooks` folder.

### Files in this repository

All the components of MioGatto is included in this repository:

* `client/` contains the node.js files for the client.
* `data/` contains all the json files, i.e, mcdict (dictionary) and anno (annotations) of the papers.
* `extra/` contains the implementation of the LLM based automation for annotations.
* `lib/` contains the project library.
* `notebooks/` contains the various jupyter notebooks used while experimenting with Open Source LLMs.
* `server/` contains the implementation of the server.
* `static/` contains the static files for the client.
* `tools/` contains our utility Python scripts.

Below is the exact same README.md of MioGatto for further reference

# MioGatto: Math Identifier-oriented Grounding Annotation Tool

## System requirements

* Python3 (3.9 or later)
* A Web Browser with MathML support (for the GUI annotation system)
    * [Firefox](https://www.mozilla.org/firefox/) is recommended

## Installation

The dependencies will be all installed with one shot:

```shell
python -m pip install -r requirements.txt
```

In case you don't want to install the dependencies into your system, please
consider using [venv](https://docs.python.org/3/library/venv.html).

## Project structure

### Files in this repository

All the components of MioGatto is included in this repository:

* `lib/` contains the project library.
* `server/` contains the implementation of the server.
* `client/` contains the implementation of the client.
* `tools/` contains our utility Python scripts.

### Files not in this repository

On the other hand, the annotation data is not included in this repository due
to the NDA constrain for [the arXMLiv dataset](https://sigmathling.kwarc.info/resources/arxmliv-dataset-2020/). The data is licensed to
[SIGMathLing members](https://sigmathling.kwarc.info/member/) as [Dataset for
Grounding of Formulae](https://sigmathling.kwarc.info/resources/grounding-dataset/).
Please consider joining [SIGMathLing](https://sigmathling.kwarc.info/member/)
to acquire the dataset.

* `arxmliv/` contains the original documents from the arXMLiv dataset
* `data/` contains the annotation data
* `sources/` contains the preprocessed documents

## Annotator's guide

For the guide with GIF animation, please refer to our Wiki:

* <https://github.com/wtsnjp/MioGatto/wiki/Annotator's-Guide>

## Prepare the input and analyze the annotated data (Advanced)

The Python scripts under the `tools` directory are mostly for the developers
for the grounding dataset. The `--help` (`-h`) option is available for all
scripts and should provide guides to their basic usages.

### Preparing data

As mentioned above, the HTML5 files in [the arXMLiv dataset](https://sigmathling.kwarc.info/resources/arxmliv-dataset-2020/)
are suitable as the input document for MioGatto. Alternatively, you can provide
the equivalent HTML5 files from LaTeX sources by using
[LaTeXML](https://dlmf.nist.gov/LaTeXML/):

```shell
latexmlc --preload=[nobibtex,ids,mathlexemes,localrawstyles]latexml.sty --format=html5 --pmml --cmml --mathtex --nodefaultresources --dest=<output HTML file> <input TeX file>
```

Then you can give the HTML5 files to our preprocess script:

```shell
python -m tools.preprocess <HTML file>
```

This will output the preprocessed HTML file to the `sources/` and generate the
initialized JSON files for the annotation to the `data/` by default. Please
refer to the help message for the options.

```shell
python -m tools.preprocess -h
```

### Analysing the annotation results

For the basic analyses for annotation data, execute:

```shell
python -m tools.analyzer <paper id>
```

Some supplemental files including graph images will be saved in the `results`
directory as default.

Similarly, analyses for the sources of grounding annotation can be performed
with the `tools.sog` script.

```shell
python -m tools.sog <paper id>
```

To calculate agreements between data by two annotators, execute:

```shell
python -m tools.agreement --target=<path to annotator's data dir> <paper id>
```

## Developing client

The client is developed with TypeScript. All development tools will be
installed with:

```shell
cd client
npm install
```

To compile the client source `client/index.ts`, execute the following in the
client directory:

```shell
npm run build
```

## Publications

* Takuto Asakura, Yusuke Miyao, Akiko Aizawa. **Building Dataset for Grounding of Formulae — Annotating Coreference Relations Among Math Identifiers**. In Proceedings of [13th Conference on Language Resources and Evaluation (LREC 2022)](https://lrec2022.lrec-conf.org). pp. 4851―4858, 2022.  
  [[paper](https://aclanthology.org/2022.lrec-1.519/)] [[bib](http://www.lrec-conf.org/proceedings/lrec2022/bib/2022.lrec-1.519.bib)] [[slides](https://speakerdeck.com/wtsnjp/lrec2022)] [[video](http://www.lrec-conf.org/proceedings/lrec2022/media/sessions/935.mp4)] [[resource](https://sigmathling.kwarc.info/resources/grounding-dataset/)]
* Takuto Asakura, Yusuke Miyao, Akiko Aizawa, Michael Kohlhase. **MioGatto: A Math Identifier-oriented Grounding Annotation Tool**. In [13th MathUI Workshop at 14th Conference on Intelligent Computer Mathematics (MathUI 2021)](https://cicm-conference.org/2021/cicm.php?event=MathUI).  
  [[preprint](https://easychair.org/publications/preprint/FSSk)] [[paper](https://kwarc.info/teaching/CICM21WS/mathui5.pdf)] [[slides](https://speakerdeck.com/wtsnjp/mathui2021)] [[code](https://github.com/wtsnjp/MioGatto)]
* Takuto Asakura, André Greiner-Petter, Akiko Aizawa, Yusuke Miyao. **Towards Grounding of Formulae**. In Proceedings of [First Workshop on Scholarly Document Processing (SDP 2020)](https://ornlcda.github.io/SDProc/). pp. 138―147, 2020.  
	[[paper](https://www.aclweb.org/anthology/2020.sdp-1.16/)] [[bib](https://www.aclweb.org/anthology/2020.sdp-1.16.bib)] [[poster](https://wtsnjp.com/posters/sdp2020-asakura-poster.pdf)]  [[resource](https://sigmathling.kwarc.info/resources/grounding-dataset/)]
* Takuto Asakura, André Greiner-Petter, Akiko Aizawa, Yusuke Miyao. **Dataset Creation for Grounding of Formulae**. In [SCIDOCA 2020](http://research.nii.ac.jp/SCIDOCA2020/).  
	[[slides](https://speakerdeck.com/wtsnjp/scidoca2020)] [[resource](https://sigmathling.kwarc.info/resources/grounding-dataset/)]

## Acknowledgements

This project has been supported by JST, ACT-X Grant Number JPMJAX2002, Japan.

## License

Copyright 2021 Takuto Asakura ([wtsnjp](https://wtsnjp.com))

This software is licensed under [the MIT license](./LICENSE).

### Third-party software

* [jQuery](https://jquery.org/): Copyright JS Foundation and other contributors. Licensed under [the MIT license](https://jquery.org/license).
* [jQuery UI](https://jqueryui.com/): Copyright jQuery Foundation and other contributors. Licensed under [the MIT license](https://github.com/jquery/jquery-ui/blob/HEAD/LICENSE.txt).

---

Takuto Asakura ([wtsnjp](https://wtsnjp.com))
