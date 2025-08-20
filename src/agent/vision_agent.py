import json
from typing import Dict, Any, Optional, List
import asyncio
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from src.agent.ollama_client import OllamaClient
from src.capture.frame_processor import FrameProcessor
from src.utils.logger import log


@dataclass
class ScreenAnalysis:
    """Result of screen analysis"""
    screen_type: str
    detected_elements: List[Dict[str, Any]]
    detected_text: List[str]
    app_name: Optional[str]
    content_playing: bool
    anomalies: Dict[str, Any]
    navigation_suggestions: Dict[str, Any]
    confidence: float
    timestamp: datetime
    raw_response: str


class VisionAgent:
    """AI agent for analyzing TV screens and making navigation decisions"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.analysis_history: List[ScreenAnalysis] = []
        self.max_history = 10
        
    async def initialize(self) -> bool:
        """Initialize the vision agent"""
        try:
            # Check if model is available
            if not await self.ollama_client.check_model():
                log.info("Model not found, pulling it...")
                if not await self.ollama_client.pull_model():
                    log.error("Failed to pull model")
                    return False
            
            log.info("Vision agent initialized successfully")
            return True
            
        except Exception as e:
            log.error(f"Error initializing vision agent: {e}")
            return False
    
    async def analyze_screen(
        self,
        frame: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ScreenAnalysis]:
        """Analyze a screen frame and return structured analysis"""
        try:
            # Check for anomalies first
            anomalies = FrameProcessor.detect_screen_anomalies(frame)
            
            # If black screen detected, return early
            if anomalies.get("black_screen"):
                return ScreenAnalysis(
                    screen_type="black_screen",
                    detected_elements=[],
                    detected_text=[],
                    app_name=None,
                    content_playing=False,
                    anomalies=anomalies,
                    navigation_suggestions={
                        "can_navigate_up": False,
                        "can_navigate_down": False,
                        "can_navigate_left": False,
                        "can_navigate_right": False,
                        "recommended_action": "wait_or_retry"
                    },
                    confidence=1.0,
                    timestamp=datetime.now(),
                    raw_response="Black screen detected"
                )
            
            # Prepare frame for AI
            processed_frame, base64_image = FrameProcessor.prepare_for_ai(frame)
            
            # Create context-aware prompt
            context = context or {}
            prompt = self.ollama_client.create_screen_analysis_prompt(context)
            
            # Analyze with AI
            result = await self.ollama_client.analyze_image(base64_image, prompt)
            
            if not result:
                log.error("Failed to get AI analysis")
                return None
            
            # Parse AI response
            analysis = self._parse_analysis_response(result["response"])
            
            # Add timing and raw response
            analysis.timestamp = datetime.now()
            analysis.raw_response = result["response"]
            
            # Update anomalies with frame processor results
            analysis.anomalies.update(anomalies)
            
            # Store in history
            self._update_history(analysis)
            
            log.info(f"Screen analyzed: {analysis.screen_type} (confidence: {analysis.confidence})")
            
            return analysis
            
        except Exception as e:
            log.error(f"Error analyzing screen: {e}")
            return None
    
    async def validate_screen(
        self,
        frame: np.ndarray,
        expected_state: str
    ) -> Dict[str, Any]:
        """Validate if screen matches expected state"""
        try:
            # Prepare frame
            _, base64_image = FrameProcessor.prepare_for_ai(frame)
            
            # Create validation prompt
            prompt = self.ollama_client.create_validation_prompt(expected_state)
            
            # Analyze
            result = await self.ollama_client.analyze_image(base64_image, prompt)
            
            if not result:
                return {
                    "matches_expected": False,
                    "actual_state": "Analysis failed",
                    "confidence": 0.0,
                    "reason": "Failed to analyze screen"
                }
            
            # Parse response
            try:
                validation = json.loads(result["response"])
                return validation
            except json.JSONDecodeError:
                # Try to extract meaningful info from text response
                response_lower = result["response"].lower()
                matches = any(word in response_lower for word in ["yes", "matches", "correct", "true"])
                
                return {
                    "matches_expected": matches,
                    "actual_state": result["response"],
                    "confidence": 0.5,
                    "reason": "Could not parse structured response"
                }
                
        except Exception as e:
            log.error(f"Error validating screen: {e}")
            return {
                "matches_expected": False,
                "actual_state": "Error",
                "confidence": 0.0,
                "reason": str(e)
            }
    
    def get_navigation_decision(
        self,
        target: str,
        current_analysis: Optional[ScreenAnalysis] = None
    ) -> Dict[str, Any]:
        """Make navigation decision based on current screen and target"""
        if not current_analysis:
            current_analysis = self.get_latest_analysis()
        
        if not current_analysis:
            return {
                "action": "wait",
                "reason": "No screen analysis available",
                "confidence": 0.0
            }
        
        # Use navigation suggestions from analysis
        suggestions = current_analysis.navigation_suggestions
        
        # Find focused element
        focused_element = None
        for element in current_analysis.detected_elements:
            if element.get("is_focused"):
                focused_element = element
                break
        
        # Decision logic based on target and current state
        decision = self._make_navigation_decision(
            target, current_analysis, focused_element, suggestions
        )
        
        return decision
    
    def _make_navigation_decision(
        self,
        target: str,
        analysis: ScreenAnalysis,
        focused_element: Optional[Dict[str, Any]],
        suggestions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal navigation decision logic"""
        target_lower = target.lower()
        
        # Check if we're already at target
        if focused_element:
            element_label = focused_element.get("label", "").lower()
            if target_lower in element_label:
                return {
                    "action": "press_ok",
                    "reason": f"Target '{target}' is currently focused",
                    "confidence": 0.9
                }
        
        # Search for target in detected elements
        target_found = False
        target_position = None
        
        for element in analysis.detected_elements:
            label = element.get("label", "").lower()
            if target_lower in label:
                target_found = True
                target_position = element.get("position", "unknown")
                break
        
        if target_found:
            # Determine navigation direction
            if "left" in target_position and suggestions.get("can_navigate_left"):
                return {
                    "action": "navigate_left",
                    "reason": f"Target '{target}' found to the left",
                    "confidence": 0.8
                }
            elif "right" in target_position and suggestions.get("can_navigate_right"):
                return {
                    "action": "navigate_right",
                    "reason": f"Target '{target}' found to the right",
                    "confidence": 0.8
                }
            elif "top" in target_position and suggestions.get("can_navigate_up"):
                return {
                    "action": "navigate_up",
                    "reason": f"Target '{target}' found above",
                    "confidence": 0.8
                }
            elif "bottom" in target_position and suggestions.get("can_navigate_down"):
                return {
                    "action": "navigate_down",
                    "reason": f"Target '{target}' found below",
                    "confidence": 0.8
                }
        
        # Use recommended action if no specific target found
        recommended = suggestions.get("recommended_action", "wait")
        return {
            "action": recommended,
            "reason": f"Following AI recommendation for screen type: {analysis.screen_type}",
            "confidence": 0.6
        }
    
    def _parse_analysis_response(self, response: str) -> ScreenAnalysis:
        """Parse AI response into ScreenAnalysis object"""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            return ScreenAnalysis(
                screen_type=data.get("screen_type", "unknown"),
                detected_elements=data.get("detected_elements", []),
                detected_text=data.get("detected_text", []),
                app_name=data.get("app_name"),
                content_playing=data.get("content_playing", False),
                anomalies=data.get("anomalies", {}),
                navigation_suggestions=data.get("navigation_suggestions", {}),
                confidence=data.get("confidence", 0.5),
                timestamp=datetime.now(),
                raw_response=response
            )
            
        except json.JSONDecodeError:
            # Fallback parsing from text
            log.warning("Could not parse JSON response, using fallback parsing")
            
            return ScreenAnalysis(
                screen_type="unknown",
                detected_elements=[],
                detected_text=[],
                app_name=None,
                content_playing=False,
                anomalies={},
                navigation_suggestions={
                    "can_navigate_up": True,
                    "can_navigate_down": True,
                    "can_navigate_left": True,
                    "can_navigate_right": True,
                    "recommended_action": "wait"
                },
                confidence=0.3,
                timestamp=datetime.now(),
                raw_response=response
            )
    
    def _update_history(self, analysis: ScreenAnalysis):
        """Update analysis history"""
        self.analysis_history.append(analysis)
        
        # Keep only recent analyses
        if len(self.analysis_history) > self.max_history:
            self.analysis_history = self.analysis_history[-self.max_history:]
    
    def get_latest_analysis(self) -> Optional[ScreenAnalysis]:
        """Get the most recent analysis"""
        return self.analysis_history[-1] if self.analysis_history else None
    
    def get_analysis_history(self) -> List[ScreenAnalysis]:
        """Get full analysis history"""
        return self.analysis_history.copy()