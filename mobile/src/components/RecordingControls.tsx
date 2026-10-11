import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from "expo-audio";
import React, { useState } from "react";
import { Alert, Pressable, StyleSheet, Text, View } from "react-native";

import { createRecordingPath, moveRecordingToLocalStorage } from "../services/recordingStore";
import { ClassRecording } from "../types";

type Props = {
  subjectId: string;
  subjectName: string;
  onRecordingStateChange: (recording: boolean) => void;
  onSaved: (recording: ClassRecording) => Promise<void>;
};

function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

export function RecordingControls({
  subjectId,
  subjectName,
  onRecordingStateChange,
  onSaved,
}: Props) {
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(recorder);
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const seconds = Math.floor(recorderState.durationMillis / 1000);

  async function startRecording() {
    try {
      const permission = await requestRecordingPermissionsAsync();
      if (!permission.granted) {
        Alert.alert(
          "Microphone permission needed",
          "Allow microphone access in device settings to record a class.",
        );
        return;
      }
      await setAudioModeAsync({
        allowsRecording: true,
        playsInSilentMode: true,
      });
      await recorder.prepareToRecordAsync();
      recorder.record();
      setIsPaused(false);
      setIsRecording(true);
      onRecordingStateChange(true);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not start recording.";
      Alert.alert("Recording could not start", message);
    }
  }

  async function togglePause() {
    try {
      if (isPaused) {
        recorder.record();
      } else {
        recorder.pause();
      }
      setIsPaused(!isPaused);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not change recording state.";
      Alert.alert("Recording error", message);
    }
  }

  async function finishRecording() {
    if (!isRecording || isSaving) {
      return;
    }
    setIsSaving(true);
    try {
      await recorder.stop();
      const sourceUri = recorder.uri;
      if (!sourceUri) {
        throw new Error("The recorder did not produce an audio file.");
      }

      const id = `${Date.now()}`;
      const localUri = createRecordingPath(id);
      await moveRecordingToLocalStorage(sourceUri, localUri);
      const saved: ClassRecording = {
        id,
        title: `${subjectName} · ${new Date().toLocaleDateString()}`,
        uri: localUri,
        createdAt: new Date().toISOString(),
        durationSeconds: seconds,
        status: "PENDING_SYNC",
        subjectId,
      };
      setIsRecording(false);
      setIsPaused(false);
      onRecordingStateChange(false);
      await setAudioModeAsync({
        allowsRecording: false,
        playsInSilentMode: true,
      });
      await onSaved(saved);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not save this recording.";
      Alert.alert("Recording could not be saved", message);
      setIsRecording(false);
      setIsPaused(false);
      onRecordingStateChange(false);
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <View style={styles.card}>
      <View style={styles.topLine}>
        <View style={[styles.statusDot, isRecording && !isPaused && styles.liveDot]} />
        <Text style={styles.statusText}>
          {isRecording ? (isPaused ? "PAUSED" : "RECORDING LIVE") : "READY TO RECORD"}
        </Text>
        {isRecording ? (
          <Text style={styles.timer}>{formatTime(seconds)}</Text>
        ) : null}
      </View>
      <View style={styles.visualizer}>
        {[14, 24, 34, 22, 42, 26, 38, 18, 30, 14, 24, 34, 20, 38, 17].map(
          (height, index) => (
            <View
              key={index}
              style={[
                styles.waveBar,
                { height },
                isRecording && !isPaused && styles.waveBarLive,
              ]}
            />
          ),
        )}
      </View>
      <Text style={styles.prompt}>
        {isRecording
          ? "Capture the important ideas. Your audio stays on this device."
          : "Record a lecture now. It will be saved on your device first."}
      </Text>
      <View style={styles.actions}>
        {!isRecording ? (
          <Pressable style={styles.recordButton} onPress={startRecording}>
            <View style={styles.recordIcon} />
            <Text style={styles.recordText}>Start recording</Text>
          </Pressable>
        ) : (
          <>
            <Pressable style={styles.pauseButton} onPress={togglePause}>
              <Text style={styles.pauseText}>{isPaused ? "Resume" : "Pause"}</Text>
            </Pressable>
            <Pressable
              style={[styles.finishButton, isSaving && styles.disabledButton]}
              onPress={finishRecording}
              disabled={isSaving}
            >
              <Text style={styles.finishText}>
                {isSaving ? "Saving…" : "Finish & save"}
              </Text>
            </Pressable>
          </>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: 22,
    borderRadius: 24,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E9E8F2",
    shadowColor: "#282447",
    shadowOpacity: 0.06,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 8 },
    elevation: 2,
  },
  topLine: { flexDirection: "row", alignItems: "center" },
  statusDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: "#B9B7C8",
    marginRight: 8,
  },
  liveDot: { backgroundColor: "#EF5C74" },
  statusText: {
    color: "#77758B",
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 1.2,
  },
  timer: {
    marginLeft: "auto",
    color: "#24213B",
    fontSize: 18,
    fontWeight: "700",
    fontVariant: ["tabular-nums"],
  },
  visualizer: {
    height: 72,
    marginTop: 19,
    marginBottom: 13,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  waveBar: {
    width: 7,
    borderRadius: 8,
    backgroundColor: "#D9D5F5",
  },
  waveBarLive: { backgroundColor: "#8B79E8" },
  prompt: { color: "#77758B", fontSize: 13, lineHeight: 20 },
  actions: {
    flexDirection: "row",
    gap: 10,
    marginTop: 20,
  },
  recordButton: {
    flex: 1,
    height: 52,
    borderRadius: 16,
    backgroundColor: "#7664D8",
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 10,
  },
  recordIcon: {
    width: 13,
    height: 13,
    borderRadius: 7,
    backgroundColor: "#FFFFFF",
  },
  recordText: { color: "#FFFFFF", fontSize: 14, fontWeight: "700" },
  pauseButton: {
    flex: 1,
    height: 50,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: "#DAD7ED",
    justifyContent: "center",
    alignItems: "center",
  },
  pauseText: { color: "#554C91", fontSize: 14, fontWeight: "700" },
  finishButton: {
    flex: 1.4,
    height: 50,
    borderRadius: 15,
    backgroundColor: "#7664D8",
    justifyContent: "center",
    alignItems: "center",
  },
  disabledButton: { opacity: 0.55 },
  finishText: { color: "#FFFFFF", fontSize: 14, fontWeight: "700" },
});
