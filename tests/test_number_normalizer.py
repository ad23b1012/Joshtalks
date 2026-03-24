"""
Tests for the Hindi Number Normalizer module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.number_normalizer import normalize_hindi_numbers


def test_simple_numbers():
    """Test simple single number word conversions."""
    assert "2" in normalize_hindi_numbers("दो")
    assert "10" in normalize_hindi_numbers("दस")
    assert "100" in normalize_hindi_numbers("सौ")
    assert "25" in normalize_hindi_numbers("पच्चीस")
    print("✓ Simple numbers passed")


def test_compound_numbers():
    """Test compound number parsing."""
    result = normalize_hindi_numbers("तीन सौ चौवन")
    assert "354" in result, f"Expected 354, got: {result}"
    
    result = normalize_hindi_numbers("एक हज़ार")
    assert "1000" in result, f"Expected 1000, got: {result}"
    
    print("✓ Compound numbers passed")


def test_idiom_preservation():
    """Test that idioms are NOT converted to numbers."""
    result = normalize_hindi_numbers("दो-चार बातें करनी हैं")
    assert "दो-चार" in result, f"Idiom should be preserved: {result}"
    
    print("✓ Idiom preservation passed")


def test_context_preservation():
    """Test that surrounding text is preserved."""
    result = normalize_hindi_numbers("मेरी उम्र पच्चीस साल है")
    assert "मेरी उम्र" in result
    assert "साल है" in result
    assert "25" in result
    print("✓ Context preservation passed")


def test_empty_and_edge():
    """Test edge cases."""
    assert normalize_hindi_numbers("") == ""
    assert normalize_hindi_numbers("कोई संख्या नहीं") == "कोई संख्या नहीं"
    print("✓ Edge cases passed")


if __name__ == "__main__":
    test_simple_numbers()
    test_compound_numbers()
    test_idiom_preservation()
    test_context_preservation()
    test_empty_and_edge()
    print("\n✅ All tests passed!")
