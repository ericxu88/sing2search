import numpy as np
import tempfile
import os
import logging
from typing import Optional, List

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.note_creation import model_output_to_notes
    BASIC_PITCH_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Basic-Pitch loaded successfully")

except ImportError as e:
    BASIC_PITCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Basic-Pitch not available: {e}")

class MIDIEmbedder:
    def __init__(self):
        #basic pitch params for optimized human voice
        self.use_basic_pitch = BASIC_PITCH_AVAILABLE
        self.onset_thresh = 0.3      
        self.frame_thresh = 0.2      
        self.infer_onsets = True
        self.min_note_len = 8        
        self.min_freq = 80        
        self.max_freq = 2000        
        self.include_pitch_bends = True
        self.multiple_pitch_bends = False
        self.melodia_trick = True
        self.midi_tempo = 120


        #multi-dim embedding
        self.pitch_weight = 0.4     
        self.interval_weight = 0.3  
        self.rhythm_weight = 0.2   
        self.contour_weight = 0.1

        logger.info("MIDI Embedder initialized")

    def audio_to_midi(self, audio_path: str):
        if not self.use_basic_pitch:
            raise Exception("Basic-Pitch not available")
        
        try:
            model_output, _, _ = predict(audio_path, ICASSP_2022_MODEL_PATH)
            midi, note_events = model_output_to_notes(
                output=model_output,
                onset_thresh=self.onset_thresh,
                frame_thresh=self.frame_thresh,
                infer_onsets=self.infer_onsets,
                min_note_len=self.min_note_len,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                include_pitch_bends=self.include_pitch_bends,
                multiple_pitch_bends=self.multiple_pitch_bends,
                melodia_trick=self.melodia_trick,
                midi_tempo=self.midi_tempo)
            return midi, note_events
        
        except Exception as e:
            logger.error(f"Audio to MIDI conversion failed: {str(e)}")
            raise

    def extract_note_features(self, midi_data, note_events) -> dict:
        features = {
            'pitches': [],
            'durations': [],
            'start_times': [],
            'velocities': []
        }

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                features['pitches'].append(note.pitch)
                features['durations'].append(note.end - note.start)
                features['start_times'].append(note.start)
                features['velocities'].append(note.velocity)

        if features['start_times']:
            sorted_indices = np.argsort(features['start_times'])
            for key in features:
                features[key] = [features[key][i] for i in sorted_indices]

        logger.debug(f"Extracted {len(features['pitches'])} notes from MIDI")
        return features
    
    def create_pitch_histogram(self, pitches: List[int]) -> np.ndarray:
        if not pitches:
            return np.zeros(128)
        
        histogram = np.zeros(128)

        for pitch in pitches:
            if 0 <= pitch <= 127:
                # gaussian smoothing added
                for offset in [-1, 0, 1]:
                    idx = pitch + offset
                    if 0 <= idx <= 127:
                        weight = 1.0 if offset == 0 else 0.1
                        histogram[idx] += weight

        return self.normalize_robust(histogram)
    
    def create_interval_histogram(self, pitches: List[int]) -> np.ndarray:
        if len(pitches) < 2:
            return np.zeros(25) 
        
        intervals = []
        for i in range(1, len(pitches)):
            interval = pitches[i] - pitches[i-1]
            interval = np.clip(interval, -12, 12)
            intervals.append(interval)

        histogram = np.zeros(25)
        for interval in intervals:
            bin_idx = interval + 12  # Shift to 0-24 range
            histogram[bin_idx] += 1

        return self.normalize_robust(histogram)
    

    def create_rhythm_features(self, start_times: List[float], durations: List[float]) -> np.ndarray:
        if len(start_times) < 2:
            return np.zeros(10)
            
        features = []
        
        # ioi is inter-onset intervals
        ioi = []
        for i in range(1, len(start_times)):
            ioi.append(start_times[i] - start_times[i-1])
        
        if ioi:
            features.extend([
                np.mean(ioi),                 
                np.std(ioi),              
                np.min(ioi),                   
                np.max(ioi),                     
                # mean is average note spacing, std is rhythm regularity, min are fastest notes and max are slowest
                len([x for x in ioi if x < 0.2]) / len(ioi)  # fraction of fast notes
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        if durations:
            features.extend([
                np.mean(durations),        
                np.std(durations),   
                np.min(durations),            
                np.max(durations), 
                # mean is avg note length, std is duration variation, min is shortest, max is longest note
                len([x for x in durations if x > 0.5]) / len(durations)  # fraction of long notes
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
            
        return np.array(features[:10]) 
    
    def create_contour_features(self, pitches: List[int]) -> np.ndarray:
        
        if len(pitches) < 3:
            return np.zeros(3)
            
        directions = []
        for i in range(1, len(pitches)):
            diff = pitches[i] - pitches[i-1]
            if diff > 0:
                directions.append(1)   
            elif diff < 0:
                directions.append(-1)  
            else:
                directions.append(0)   
        
        if not directions:
            return np.zeros(3)
            
        # direction patterns
        up_count = sum(1 for d in directions if d == 1)
        down_count = sum(1 for d in directions if d == -1)
        same_count = sum(1 for d in directions if d == 0)
        
        total = len(directions)
        contour = np.array([
            up_count / total,   
            down_count / total, 
            same_count / total   
        ])
        
        return contour
    
    def normalize_robust(self, vector: np.ndarray) -> np.ndarray:
        
        vector = np.array(vector, dtype=np.float32)
        
        # L2 normalization with small epsilon to avoid zero division
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            return vector / norm
        else:
            return vector
        
    def generate_embedding(self, audio_path: str) -> Optional[np.ndarray]:
    
        try:
            if not self.use_basic_pitch:
                logger.error("Basic-Pitch not available")
                return None
            
            logger.info(f"Generating embedding for: {audio_path}")
            
            # audio -> midi
            midi_data, note_events = self.audio_to_midi(audio_path)
            
            # extract features
            note_features = self.extract_note_features(midi_data, note_events)
            
            if not note_features['pitches']:
                logger.warning("No notes detected: returning zero embedding")
                return np.zeros(128, dtype=np.float32)
            
            #multi-dim features
            
            # pitch histogram (piano range)
            pitch_hist = self.create_pitch_histogram(note_features['pitches'])
            pitch_hist = pitch_hist[21:109]  # A0 to C8 (piano range)
            
            # interval histogram for pitch invariance
            interval_hist = self.create_interval_histogram(note_features['pitches'])
            
            # rhythm features
            rhythm_features = self.create_rhythm_features(
                note_features['start_times'], 
                note_features['durations']
            )
            
            # contour features
            contour_features = self.create_contour_features(note_features['pitches'])
            
            # padding for 128 dim
            overall_features = np.array([
                len(note_features['pitches']) / 50.0,  
                np.std(note_features['pitches']) / 12.0 if len(note_features['pitches']) > 1 else 0  
            ])
            
            # combine all features with weights
            embedding = np.concatenate([
                pitch_hist * self.pitch_weight,     
                interval_hist * self.interval_weight, 
                rhythm_features * self.rhythm_weight, 
                contour_features * self.contour_weight, 
                overall_features * 0.1               
            ])
            
            # normalize again
            embedding = self.normalize_robust(embedding)
            
            logger.info(f"Generated embedding: {embedding.shape} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None
        
    def generate_embedding_from_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> Optional[np.ndarray]:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            try:
                return self.generate_embedding(tmp_path)
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Failed to generate embedding from bytes: {str(e)}")
            return None

    def get_embedding_info(self) -> dict:
        return {
            "model_type": "MIDI Embedder",
            "dimensions": 128,
            "basic_pitch_available": self.use_basic_pitch,
            "feature_breakdown": {
                "pitch_histogram": f"88 dims (weight: {self.pitch_weight})",
                "interval_histogram": f"25 dims (weight: {self.interval_weight})", 
                "rhythm_features": f"10 dims (weight: {self.rhythm_weight})",
                "contour_features": f"3 dims (weight: {self.contour_weight})",
                "overall_features": "2 dims"
            }
        }
