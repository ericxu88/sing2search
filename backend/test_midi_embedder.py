import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from app.models.midi_embedder import MIDIEmbedder

def create_test_audio():
    sr = 22050
    # C major scale frequencies
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    
    audio = np.array([])
    for freq in freqs:
        t = np.linspace(0, 0.5, int(sr * 0.5))  # 0.5 seconds per note
        note = 0.3 * np.sin(2 * np.pi * freq * t)
        
        # add fade in/out
        fade_samples = int(0.05 * sr)
        note[:fade_samples] *= np.linspace(0, 1, fade_samples)
        note[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        audio = np.concatenate([audio, note])
    
    return audio, sr

def test_embedder():
    
    print("Testing Your MIDI Embedder")
    
    embedder = MIDIEmbedder()
    
    info = embedder.get_embedding_info()
    print(f"Model: {info['model_type']}")
    print(f"Dimensions: {info['dimensions']}")
    print(f"Basic-Pitch Available: {info['basic_pitch_available']}")
    print()
    
    if not info['basic_pitch_available']:
        print("Basic-Pitch not available")
        return False
    
    print("Feature Breakdown:")
    for feature, desc in info['feature_breakdown'].items():
        print(f"   • {feature}: {desc}")
    print()
    
    print("Creating test audio (C major scale)...")
    audio, sr = create_test_audio()
    
    #save to temp
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sr)
        audio_path = tmp_file.name
    
    try:
        print("Generating embedding...")
        embedding = embedder.generate_embedding(audio_path)
        
        if embedding is not None:
            print("Embedding generated successfully!")
            print(f"   Shape: {embedding.shape}")
            print(f"   Data type: {embedding.dtype}")
            print(f"   Non-zero elements: {np.count_nonzero(embedding)}/128")
            print(f"   L2 norm: {np.linalg.norm(embedding):.6f}")
            print(f"   Max value: {np.max(embedding):.6f}")
            print(f"   Min value: {np.min(embedding):.6f}")
            
            top_indices = np.argsort(embedding)[-10:][::-1]
            print(f"   🔝 Top 5 feature indices: {top_indices[:5]}")
            print(f"   🔝 Top 5 values: {embedding[top_indices[:5]]}")
            
            return embedding
        else:
            print("Failed to generate embedding")
            return None
            
    finally:
        os.unlink(audio_path)

def test_pinecone_integration(embedding):
    
    print("\n Testing Pinecone Integration")
    
    try:
        from app.core.database import PineconeClient
        
        db_client = PineconeClient()
        
        if not db_client.is_connected():
            print("Pinecone not connected")
            return False
        
        print("Storing test embedding")
        
        test_metadata = {
            "title": "Test C Major Scale",
            "artist": "Embedder Test",
            "type": "test_melody"
        }
        
        success = db_client.upsert_embedding(
            song_id="test-your-embedder",
            embedding=embedding,
            metadata=test_metadata,
            namespace="test"
        )
        
        if success:
            print("Embedding stored successfully")
            
            print("Searching for the embedding...")
            results = db_client.search_similar(
                embedding=embedding,
                top_k=1,
                namespace="test"
            )
            
            if results:
                result = results[0]
                print(f"Found match!")
                print(f"ID: {result['id']}")
                print(f"Score: {result['score']:.6f}")
                print(f"Title: {result['metadata']['title']}")
                return True
            else:
                print("No search results found")
                return False
        else:
            print("Failed to store embedding")
            return False
            
    except Exception as e:
        print(f"Pinecone test failed: {str(e)}")
        return False

def test_consistency():
    
    print("\n Testing Consistency")
    
    embedder = MIDIEmbedder()
    audio, sr = create_test_audio()
    
    # generate embedding multiple times
    embeddings = []
    
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            audio_path = tmp_file.name
        
        try:
            embedding = embedder.generate_embedding(audio_path)
            if embedding is not None:
                embeddings.append(embedding)
        finally:
            os.unlink(audio_path)
    
    if len(embeddings) >= 2:
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        avg_sim = np.mean(similarities)
        print(f"Average self-similarity: {avg_sim:.6f}")
        
        if avg_sim > 0.999:
            print("Perfect consistency")
        elif avg_sim > 0.99:
            print("Excellent consistency")
        else:
            print("Some variation detected")
        
        return True
    else:
        print("Could not generate enough embeddings")
        return False

if __name__ == "__main__":
    print("MIDI Embedder Test")
    
    embedding = test_embedder()
    
    if embedding is not None:
        pinecone_success = test_pinecone_integration(embedding)
        
        consistency_success = test_consistency()
        
        print("TEST SUMMARY")
        
        if pinecone_success and consistency_success:
            print("All tests passed!")
            print("Embedder working correctly")
            print("Pinecone integration working")
            print("Consistent results")
        else:
            print(f"Pinecone: {'GOOD' if pinecone_success else 'BAD'}")
            print(f"Consistency: {'GOOD' if consistency_success else 'BAD'}")
    else:
        print("\n Embedder test failed")