Each result version corespond to the snapshot code of that version.

Tipically base model were trained with batch size 32, no gradient accumulation, no gradient cliping, with enabled batch normalizations (if no bug occures), no oversample and precomputed weighted loss.
The large model was trained with batch size 16, gradient accumulation of 2, gradient cliping, disabled batch normalization, with oversample up to 1000 and weighted loss.

For version 2 of the base model, we also used the oversample of 1000 combined with weighted loss.