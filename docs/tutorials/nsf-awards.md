# One space, many kinds of things: NSF awards

The U.S. National Science Foundation publishes every award it makes, each with an
abstract written for the public, and states that this data is in the public domain. An
award also says who leads it, at which institution, in which state, and which program,
division and directorate paid for it.

That makes it a good illustration of the central idea. Every one of those descriptions
becomes a *factor*, and every factor gets its own set of points in the same space as
the words and the projects. You can then ask which programs are closest to a topic,
which institutions resemble each other, or which programs sit near a state.

The library built on this page is also online as a [live demo](../../demo/).

## Build it

Fiscal year 2024 comes with the repository, already prepared, as
`nsf-data/nsf-awards-2024.jsonl`: one project per line, 32 MB. Build a library from it
like any other file:

```bash
sl init libraries/nsf
sl add libraries/nsf nsf-data/nsf-awards-2024.jsonl --text title,abstract --id id \
    --factor investigator --factor institution --factor state --factor program \
    --factor division --factor directorate --factor year
sl build libraries/nsf
```

The script `examples/nsf_awards.py` does the same and then runs the searches shown
below. On a laptop with eight cores in use, the build took about three minutes.

```bash
python examples/nsf_awards.py
```

```
fiscal year 2024: 9932 projects
built in 173.8s
{'document': 9932, 'word': 47603, 'directorate': 11, 'division': 80, 'institution': 1713,
 'investigator': 17499, 'program': 491, 'state': 57, 'year': 1}
```

The file was prepared from NSF's own download by the same script. Three things were
done to the raw data:

- The year holds 11,687 awards. The parts of a collaborative project are separate
  awards with the same abstract, so awards that share an abstract were merged into one
  document carrying all their investigators and institutions. That leaves 9,932
  projects.
- NSF ends every abstract with the same sentence about its review criteria. It was
  removed, since it says nothing about any one project.
- Only the title, abstract, names of people and organizations, amount, type and start
  date were kept. NSF's files also list e-mail addresses and telephone numbers, which
  were left out.

To use other years, name them: `python examples/nsf_awards.py libraries/nsf3 2022 2023 2024`.
The script downloads each year from NSF (50 to 160 MB) into `nsf-data/` and prepares it
the same way.

## From a topic to everything else

The script ends by ranking one query, "sea ice and polar ecosystems", against several
kinds of thing. Its output is shown here. First the projects:

```
  1  0.814  RAPID: The Collapsing Ice and Ocean Ecosystem of Milne Fiord, Canada
  2  0.810  Postdoctoral Fellowship: OPP-PRF: Constraining Potential Impacts of Brine Firn A
  3  0.810  Postdoctoral Fellowship: OPP-PRF: Constraining the Impacts of Snowfall and Storm
  4  0.808  How much and why did Ice Sheets melt during the Last Interglacial (HISEAS)
  5  0.804  Doctoral Dissertation Research: Using Novel Luminescence Dating Techniques to Co
```

Then the funding programs:

```
  1  0.769  ANT Ocean & Atmos Sciences
  2  0.764  POST DOC/TRAVEL
  3  0.750  ANT Glaciology
  4  0.746  ANS-Arctic Natural Sciences
  5  0.741  ANT Earth Sciences
```

("ANT" is Antarctic.) And the words:

```
  1  0.750  ice
  2  0.746  polar
  3  0.728  sea
  4  0.670  ecosystems
  5  0.642  ocean
```

From the command line the same searches are
`sl search libraries/nsf "sea ice and polar ecosystems"`, adding `--space program` or
`--space word`.

A query that mixes two fields lands between them. Here are the divisions closest to
"machine learning for protein structure":

```bash
sl search libraries/nsf "machine learning for protein structure" --space division -k 4
```

```
   1   0.8429  Computing & AI Foundations
   2   0.8356  Molecular Biosciences
   3   0.8348  Innovation Tools, Informatics, & Functional Genomics
   4   0.8324  Civil, Mechanical, & Manufacturing Innovation
```

## From one thing to others like it

Institutions that do similar work to an oceanographic institute:

```bash
sl similar libraries/nsf "Woods Hole Oceanographic Institution" --space institution -k 5
```

```
   1   1.0000  Woods Hole Oceanographic Institution
   2   0.9985  University of California-San Diego Scripps Inst of Oceanography
   3   0.9961  University of Alaska Fairbanks Campus
   4   0.9954  Texas A&M University Corpus Christi
   5   0.9949  University of North Carolina at Wilmington
```

Programs near the Linguistics program:

```bash
sl similar libraries/nsf "Linguistics" --space program -k 5
```

```
   1   1.0000  Linguistics
   2   0.9937  DS -Developmental Sciences
   3   0.9935  DLI-Dyn Language Infrastructur
   4   0.9934  DDRI Linguistics
   5   0.9922  Perception, Action & Cognition
```

## Across kinds

`--target` ranks a different kind of thing from the one you start with. The programs
closest to an institution:

```bash
sl similar libraries/nsf "Woods Hole Oceanographic Institution" --space institution --target program -k 5
```

```
   1   0.9966  Marine Geology and Geophysics
   2   0.9964  OCE Postdoctoral Fellowships
   3   0.9963  ANT Organisms & Ecosystems
   4   0.9958  Geomorphology & Land-use Dynam
   5   0.9958  PHYSICAL OCEANOGRAPHY
```

And the programs closest to a state:

```bash
sl similar libraries/nsf "Alaska" --space state --target program -k 5
```

```
   1   0.9973  XC-Crosscutting Activities Pro
   2   0.9953  OCE-Ocean Sciences Research
   3   0.9946  Geomorphology & Land-use Dynam
   4   0.9944  Archaeology
   5   0.9943  EDUCATION AND HUMAN RESOURCES
```

## Reading the numbers

Three things in these results are worth understanding before you trust them.

**Big things all look alike.** An institution or program is the sum of many projects,
and every NSF abstract shares a good deal of language about research, students and
impact. Sums of many abstracts therefore point in nearly the same direction, which is
why the similarities above are all near 0.99. The order is still informative; the size
of the gaps is not. Compare ranks within one list, and do not read 0.996 as "almost
identical".

**Small things are noisy.** An institution with a single award is just that one
project. Ranking institutions against a query therefore puts one-award institutions
first whenever their one project matches well, as the script's last list shows:

```
institutions closest to: sea ice and polar ecosystems
  1  0.760  Planetary Science Institute
  2  0.742  Health Research Incorporated/New York State Department of Health
  3  0.719  Hankel, Camille
```

(Fellowships paid to a person list that person as the institution.) Going from a query
to programs or divisions, which each hold many projects, works better than going to
institutions or investigators.

**From a big thing to words is vague.** The words closest to an institution are the
generic words of grant writing, because that shared language dominates the sum:

```bash
sl similar libraries/nsf "Woods Hole Oceanographic Institution" --space institution --target word -k 5
```

```
   1   0.8072  design
   2   0.8039  process
   3   0.7980  study
   4   0.7952  use
   5   0.7905  current
```

To characterize an institution, look at its nearest programs instead.

## In the browser

```bash
sl serve libraries/nsf --open
```

"Find similar" lets you start from a program, an institution, a state or an
investigator and show any other kind of thing. Each project's details link to its page
on nsf.gov. The export is 114 MB.
