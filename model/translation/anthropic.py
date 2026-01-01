import anthropic
import os
from typing import Dict, Optional
import json

class EdoPeriodTranslationPipeline:
    """
    Complete pipeline for translating Edo period Japanese texts to English.
    Pipeline: Image → Transcription → Modern Japanese → English
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the pipeline with Anthropic API.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env variable
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY env variable or pass api_key parameter")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"  # Latest Claude Sonnet
    
    def transcribe_image(self, model, image_path: str) -> str:
        """
        Step 1: Transcribe classical Japanese from image.
        Replace this with your trained model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Transcribed classical Japanese text
        """
        # TODO: Replace with your actual transcription model
        # For now, this is a placeholder
        print(f"[Transcription] Processing image: {image_path}")
        
        # Example: return your_model.transcribe(image_path)
        # Placeholder return for testing
        # return "古文の例文がここに入ります"
        return model.transcribe(image_path)
    
    def classical_to_modern(self, classical_text: str) -> Dict[str, str]:
        """
        Step 2: Convert classical/Edo-period Japanese to modern Japanese.
        
        Args:
            classical_text: Classical Japanese text from transcription
            
        Returns:
            Dict with 'modern_japanese' and 'notes' (any ambiguities)
        """
        print(f"[Classical→Modern] Converting: {classical_text[:50]}...")
        
        prompt = f"""You are an expert in classical Japanese literature, specifically Edo period texts (1603-1868).

Convert the following classical Japanese text to modern Japanese (現代日本語).

Requirements:
- Preserve the original meaning and nuance precisely
- Convert classical grammar forms to modern equivalents
- Update archaic vocabulary to contemporary terms
- Maintain the tone and register appropriate to the original context
- If there are ambiguous passages, note them

Classical Japanese text:
{classical_text}

Respond in JSON format:
{{
    "modern_japanese": "the converted modern Japanese text",
    "notes": "any ambiguities or important conversion decisions (can be empty string if none)"
}}"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract response
            response_text = message.content[0].text
            
            # Parse JSON response
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            print(f"[Classical→Modern] Success: {result['modern_japanese'][:50]}...")
            return result
            
        except Exception as e:
            print(f"[Classical→Modern] Error: {str(e)}")
            raise
    
    def translate_to_english(self, modern_japanese: str, classical_text: str = "") -> Dict[str, str]:
        """
        Step 3: Translate modern Japanese to English.
        
        Args:
            modern_japanese: Modern Japanese text from previous step
            classical_text: Original classical text for context (optional)
            
        Returns:
            Dict with 'english_translation' and 'translation_notes'
        """
        print(f"[Modern→English] Translating: {modern_japanese[:50]}...")
        
        context = f"\n\nOriginal classical text for context:\n{classical_text}" if classical_text else ""
        
        prompt = f"""Translate the following modern Japanese text to natural, fluent English.

This text was originally from an Edo period Japanese document and has been converted to modern Japanese. Provide a translation that:
- Reads naturally in English
- Preserves the meaning and nuance
- Maintains appropriate formality level
- Notes any cultural context that may be important for understanding

Modern Japanese text:
{modern_japanese}{context}

Respond in JSON format:
{{
    "english_translation": "the English translation",
    "translation_notes": "any important cultural context or translation decisions (can be empty string if none)"
}}"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract response
            response_text = message.content[0].text
            
            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            print(f"[Modern→English] Success: {result['english_translation'][:50]}...")
            return result
            
        except Exception as e:
            print(f"[Modern→English] Error: {str(e)}")
            raise
    
    def process(self, model, image_path: str) -> Dict:
        """
        Complete pipeline: Image → Classical → Modern → English
        
        Args:
            image_path: Path to image file with classical Japanese text
            
        Returns:
            Dict containing all intermediate results and final translation
        """
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}\n")
        
        # Step 1: Transcribe
        classical_text = self.transcribe_image(model, image_path)
        
        # Step 2: Classical to Modern
        modern_result = self.classical_to_modern(classical_text)
        
        # Step 3: Modern to English
        translation_result = self.translate_to_english(
            modern_result['modern_japanese'],
            classical_text
        )
        
        # Compile results
        final_result = {
            'classical_japanese': classical_text,
            'modern_japanese': modern_result['modern_japanese'],
            'conversion_notes': modern_result['notes'],
            'english_translation': translation_result['english_translation'],
            'translation_notes': translation_result['translation_notes']
        }
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}\n")
        
        return final_result


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = EdoPeriodTranslationPipeline()
    
    # Example with a direct classical text (bypass transcription for testing)
    classical_sample = "かくて年月を経て、遂に都へ上りけり"
    
    print("Testing with sample classical text...")
    print(f"Input: {classical_sample}\n")
    
    # Test classical to modern conversion
    modern_result = pipeline.classical_to_modern(classical_sample)
    print(f"\nModern Japanese: {modern_result['modern_japanese']}")
    if modern_result['notes']:
        print(f"Notes: {modern_result['notes']}")
    
    # Test translation
    translation_result = pipeline.translate_to_english(
        modern_result['modern_japanese'],
        classical_sample
    )
    print(f"\nEnglish: {translation_result['english_translation']}")
    if translation_result['translation_notes']:
        print(f"Translation Notes: {translation_result['translation_notes']}")
    
    print("\n" + "="*60)
    print("To use with images, replace transcribe_image() with your model")
    print("="*60)