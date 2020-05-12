# bachelor-thesis
The code for my bachelor thesis, which explores neural network-based models to generate solutions for Raven's Progressive Matrices.

## notes on the data generation script
1. The script can only generate PGMs in the non-distractive setting.
2. Up to 6 rule triples can be present in the same matrix. When generating the actual dataset, I can control up to how many triples will be present.
3. Similarly, the script has methods to enable the generation of all kinds of different sub-datasets. For instance, one can sample arbitrarily many PGMs that each satisfy the same combination of triples; certain triples can be excluded from the dataset, etc. This should make it easier to generate datasets that measure different generalization regimes down the road.
4. In its current form, the code is not camera-ready: there are print statements to erase and so on, but the structure of it is done.
5. It takes less than 30 minutes to generate a million PGMs, excluding the time to actually save each one of them, which I have yet to measure.
6. Certain triple combinations, e.g. (shape, size, OR) & (shape, position, AND) are not yet implemented, although they should be mathematically conceivable. The code is written in such a way that adding this functionality in the future won't be a very big pain if we need it.
7. Color and size attributes are often very hard to distinguish; e.g.: the human eye can't tell apart the shade of black encoded by 0 and the shape of black encoded by 1. This should not be a problem at this stage where we work with object-level representations, but might pose challenges in the future. I thought of a workaround to this problem which coarsens the size/color domains.
