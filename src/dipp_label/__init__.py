"""DiPP-Label: portable, training-fold-local plasma proteomic models."""
__version__ = "0.1.0"

# Keep the source experiment's probability-column order explicit everywhere.
CLASSES = ("HC", "Prodromal", "PD")
PAIRS = ((2, 1), (0, 1), (0, 2))
