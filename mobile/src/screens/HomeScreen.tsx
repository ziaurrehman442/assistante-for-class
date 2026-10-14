import NetInfo, { NetInfoState } from "@react-native-community/netinfo";
import {
  setAudioModeAsync,
  useAudioPlayer,
  useAudioPlayerStatus,
} from "expo-audio";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import { RecordingControls } from "../components/RecordingControls";
import {
  createSubjectFolder,
  loadRecordings,
  loadSubjectFolders,
  saveRecordings,
} from "../services/recordingStore";
import {
  forgetSpeakerProfile,
  renameSpeakerAndRegenerate,
} from "../services/speakerService";
import { syncPendingRecordings } from "../services/syncService";
import { ClassRecording, RecordingSpeaker, SubjectFolder } from "../types";

type TabName = "Transcript" | "Detailed Notes" | "Easy Notes" | "Revision Quiz";

const tabs: TabName[] = [
  "Transcript",
  "Detailed Notes",
  "Easy Notes",
  "Revision Quiz",
];

function formatDuration(seconds: number): string {
  return `${Math.floor(seconds / 60)} min ${seconds % 60} sec`;
}

export function HomeScreen() {
  const [recordings, setRecordings] = useState<ClassRecording[]>([]);
  const [subjectFolders, setSubjectFolders] = useState<SubjectFolder[]>([]);
  const [selectedSubjectId, setSelectedSubjectId] = useState<string | null>(null);
  const [newSubjectName, setNewSubjectName] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<TabName>("Detailed Notes");
  const [isOnline, setIsOnline] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [loadedPlayerId, setLoadedPlayerId] = useState<string | null>(null);
  const [speakerDrafts, setSpeakerDrafts] = useState<Record<string, string>>({});
  const [renamingSpeakerId, setRenamingSpeakerId] = useState<string | null>(null);
  const player = useAudioPlayer(null, { updateInterval: 250 });
  const playerStatus = useAudioPlayerStatus(player);

  const refreshRecordings = useCallback(async () => {
    try {
      const folders = await loadSubjectFolders();
      const localRecordings = await loadRecordings();
      setSubjectFolders(folders);
      setRecordings(localRecordings);
      setSelectedSubjectId((currentId) =>
        currentId && folders.some((folder) => folder.id === currentId)
          ? currentId
          : folders[0]?.id ?? null,
      );
      setLoadError(null);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not read saved recordings.";
      setLoadError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const selectedSubject = subjectFolders.find(
    (folder) => folder.id === selectedSubjectId,
  );
  const subjectRecordings = recordings.filter(
    (recording) => recording.subjectId === selectedSubjectId,
  );

  const isSyncing = useMemo(
    () => recordings.some((recording) => recording.status === "SYNCING"),
    [recordings],
  );

  // The sync loop persists progress to storage as each poll lands, but this
  // screen holds recordings in React state, so it has to re-read while a job is
  // in flight. Without this the progress bar only appeared after the sync
  // finished. The timer is only mounted while something is actually processing.
  useEffect(() => {
    if (!isSyncing) {
      return;
    }
    const id = setInterval(() => void refreshRecordings(), 2000);
    return () => clearInterval(id);
  }, [isSyncing, refreshRecordings]);

  async function addSubjectFolder() {
    try {
      const folder = await createSubjectFolder(newSubjectName);
      const folders = await loadSubjectFolders();
      setSubjectFolders(folders);
      setSelectedSubjectId(folder.id);
      setNewSubjectName("");
      setSelectedId(null);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not create this subject folder.";
      Alert.alert("Could not create folder", message);
    }
  }

  function selectSubject(folder: SubjectFolder) {
    if (isRecording) {
      return;
    }
    setSelectedSubjectId(folder.id);
    setSelectedId(null);
  }

  useEffect(() => {
    void refreshRecordings();
    const unsubscribe = NetInfo.addEventListener((state: NetInfoState) => {
      const connected = state.isConnected === true;
      setIsOnline(connected);
      if (connected) {
        void syncPendingRecordings()
          .then(refreshRecordings)
          .catch((error: unknown) => {
            const message =
              error instanceof Error ? error.message : "Could not sync recordings.";
            console.warn("Audio queue could not be synced:", message);
          });
      }
    });
    return unsubscribe;
  }, [refreshRecordings]);

  useEffect(() => {
    if (playerStatus.didJustFinish) {
      setPlayingId(null);
    }
  }, [playerStatus.didJustFinish]);

  async function saveRecording(recording: ClassRecording) {
    try {
      const updated = [recording, ...recordings];
      await saveRecordings(updated);
      setRecordings(updated);
      setSelectedId(recording.id);
      if (isOnline) {
        await syncPendingRecordings();
        await refreshRecordings();
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not save this recording.";
      Alert.alert("Could not save class", message);
    }
  }

  async function reprocessRecording(recording: ClassRecording) {
    try {
      const latest = await loadRecordings();
      const updated = latest.map((item) =>
        item.id === recording.id
          ? {
              ...item,
              status: "PENDING_SYNC" as const,
              processingJobId: undefined,
              lastError: undefined,
              // Clear the old results so stale notes are never mistaken for
              // the outcome of the re-run.
              transcript: undefined,
              dialogue: undefined,
              speakers: undefined,
              materials: undefined,
              attempt: (item.attempt ?? 1) + 1,
            }
          : item,
      );
      await saveRecordings(updated);
      setRecordings(updated);
      setSpeakerDrafts({});
      if (isOnline) {
        await syncPendingRecordings();
        await refreshRecordings();
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not re-process this recording.";
      Alert.alert("Could not re-process", message);
    }
  }

  async function togglePlayback(recording: ClassRecording) {
    try {
      if (playingId === recording.id && playerStatus.playing) {
        player.pause();
        setPlayingId(null);
        return;
      }
      if (loadedPlayerId === recording.id) {
        if (playerStatus.didJustFinish) {
          await player.seekTo(0);
        }
        player.play();
        setPlayingId(recording.id);
        return;
      }
      await setAudioModeAsync({
        allowsRecording: false,
        playsInSilentMode: true,
      });
      player.replace({ uri: recording.uri });
      player.play();
      setLoadedPlayerId(recording.id);
      setPlayingId(recording.id);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not play this recording.";
      Alert.alert("Playback error", message);
      setPlayingId(null);
    }
  }

  async function renameSpeaker(
    recording: ClassRecording,
    speaker: RecordingSpeaker,
  ) {
    const displayName = speakerDrafts[speaker.id]?.trim();
    if (!displayName) {
      Alert.alert("Name required", "Enter a speaker name before saving.");
      return;
    }
    if (!recording.dialogue) {
      Alert.alert("Transcript unavailable", "Sync this class again before renaming speakers.");
      return;
    }

    setRenamingSpeakerId(speaker.id);
    try {
      const result = await renameSpeakerAndRegenerate(
        speaker.id,
        displayName,
        recording.dialogue,
      );
      const updatedRecording: ClassRecording = {
        ...recording,
        transcript: result.transcript,
        dialogue: result.dialogue,
        speakers: recording.speakers?.map((item) =>
          item.id === speaker.id
            ? { ...item, display_name: displayName, is_named: true }
            : item,
        ),
        materials: result.materials,
        lastError: undefined,
      };
      const updatedRecordings = recordings.map((item) =>
        item.id === recording.id ? updatedRecording : item,
      );
      await saveRecordings(updatedRecordings);
      setRecordings(updatedRecordings);
      setSpeakerDrafts((drafts) => ({ ...drafts, [speaker.id]: displayName }));
      Alert.alert(
        "Speaker saved",
        `${displayName} is saved for future recordings from this backend.`,
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not rename this speaker.";
      Alert.alert("Could not regenerate notes", message);
    } finally {
      setRenamingSpeakerId(null);
    }
  }

  function confirmForgetSpeaker(recording: ClassRecording, speaker: RecordingSpeaker) {
    Alert.alert(
      "Forget this voice?",
      "The backend will delete this local voice profile. Existing transcripts will keep their current speaker names.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Forget voice",
          style: "destructive",
          onPress: () => {
            void forgetSpeakerProfile(speaker.id)
              .then(async () => {
                const updatedRecordings = recordings.map((item) =>
                  item.id === recording.id
                    ? {
                        ...item,
                        speakers: item.speakers?.filter(
                          (profile) => profile.id !== speaker.id,
                        ),
                      }
                    : item,
                );
                await saveRecordings(updatedRecordings);
                setRecordings(updatedRecordings);
                setSpeakerDrafts((drafts) => {
                  const remaining = { ...drafts };
                  delete remaining[speaker.id];
                  return remaining;
                });
              })
              .catch((error: unknown) => {
                const message =
                  error instanceof Error ? error.message : "Could not delete this voice profile.";
                Alert.alert("Could not forget voice", message);
              });
          },
        },
      ],
    );
  }

  function renderResults(recording: ClassRecording) {
    if (selectedTab === "Transcript") {
      return (
        <View style={styles.resultsContent}>
          <Text style={styles.transcriptHelp}>
            Rename each voice once. Its voice profile is shared across subject folders and used to label that speaker in future classes.
          </Text>
          {(recording.speakers ?? []).map((speaker) => (
            <View key={speaker.id} style={styles.speakerEditor}>
              <TextInput
                accessibilityLabel={`Name for ${speaker.display_name}`}
                value={speakerDrafts[speaker.id] ?? speaker.display_name}
                onChangeText={(value) =>
                  setSpeakerDrafts((drafts) => ({ ...drafts, [speaker.id]: value }))
                }
                placeholder="Teacher or Student 1"
                placeholderTextColor="#AAA7B8"
                maxLength={40}
                style={styles.speakerInput}
              />
              <Pressable
                style={[
                  styles.renameButton,
                  (!isOnline || renamingSpeakerId !== null) && styles.renameButtonDisabled,
                ]}
                disabled={!isOnline || renamingSpeakerId !== null}
                onPress={() => void renameSpeaker(recording, speaker)}
              >
                <Text style={styles.renameButtonText}>
                  {renamingSpeakerId === speaker.id
                    ? "Generating…"
                    : "Save & regenerate"}
                </Text>
              </Pressable>
              {speaker.is_named ? (
                <Pressable
                  style={[
                    styles.forgetButton,
                    (!isOnline || renamingSpeakerId !== null) && styles.renameButtonDisabled,
                  ]}
                  disabled={!isOnline || renamingSpeakerId !== null}
                  onPress={() => confirmForgetSpeaker(recording, speaker)}
                >
                  <Text style={styles.forgetButtonText}>Forget voice profile</Text>
                </Pressable>
              ) : null}
            </View>
          ))}
          {(recording.dialogue ?? []).map((turn, index) => (
            <View key={`${turn.speaker_id}-${turn.start}-${index}`} style={styles.transcriptTurn}>
              <View style={styles.transcriptTurnHeader}>
                <Text style={styles.transcriptSpeaker}>{turn.speaker_name}</Text>
                <Text style={styles.transcriptTime}>
                  {`${Math.floor(turn.start / 60).toString().padStart(2, "0")}:${Math.floor(turn.start % 60).toString().padStart(2, "0")}`}
                </Text>
              </View>
              <Text style={styles.transcriptText}>{turn.text}</Text>
              {turn.overlapping_speakers?.length ? (
                <Text style={styles.interruptionLabel}>
                  Overlapping: {turn.overlapping_speakers.join(", ")}
                </Text>
              ) : null}
            </View>
          ))}
          {recording.dialogue?.length === 0 && recording.transcript ? (
            <Text style={styles.transcriptText}>{recording.transcript}</Text>
          ) : null}
        </View>
      );
    }
    if (!recording.materials) {
      return null;
    }

    if (selectedTab === "Detailed Notes") {
      const notes = recording.materials.detailed_class_notes;
      return (
        <View style={styles.resultsContent}>
          <Text style={styles.resultHeading}>Main topics</Text>
          {notes.main_topics.map((item, index) => (
            <Text key={`topic-${index}`} style={styles.bullet}>•  {item}</Text>
          ))}
          <Text style={styles.resultHeading}>Definitions</Text>
          {notes.definitions.map((item, index) => (
            <Text key={`definition-${index}`} style={styles.bullet}>•  {item}</Text>
          ))}
          <Text style={styles.resultHeading}>Teacher explanations</Text>
          {notes.teacher_explanations.map((item, index) => (
            <Text key={`explanation-${index}`} style={styles.bullet}>•  {item}</Text>
          ))}
          <Text style={styles.resultHeading}>Questions & answers</Text>
          {notes.qa.map((item, index) => (
            <View key={`qa-${index}`} style={styles.qaItem}>
              <Text style={styles.qaQuestion}>{item.question}</Text>
              <Text style={styles.bullet}>{item.answer}</Text>
            </View>
          ))}
        </View>
      );
    }

    if (selectedTab === "Easy Notes") {
      return (
        <View style={styles.resultsContent}>
          {recording.materials.easy_notes.map((item, index) => (
            <Text key={`easy-${index}`} style={styles.bullet}>•  {item}</Text>
          ))}
        </View>
      );
    }

    return (
      <View style={styles.resultsContent}>
        {recording.materials.revision_quiz.mcqs.map((question, index) => (
          <View key={`mcq-${index}`} style={styles.quizItem}>
            <Text style={styles.qaQuestion}>{index + 1}. {question.question}</Text>
            {question.options.map((option, optionIndex) => (
              <Text key={`option-${optionIndex}`} style={styles.bullet}>
                {String.fromCharCode(65 + optionIndex)}. {option}
              </Text>
            ))}
            <Text style={styles.answer}>Answer: {question.answer}</Text>
          </View>
        ))}
        <Text style={styles.resultHeading}>Conceptual questions</Text>
        {recording.materials.revision_quiz.conceptual_questions.map((question, index) => (
          <Text key={`concept-${index}`} style={styles.bullet}>
            {index + 1}. {question}
          </Text>
        ))}
      </View>
    );
  }

  function renderRecording({ item }: { item: ClassRecording }) {
    const selected = item.id === selectedId;
    const statusLabel =
      item.status === "SYNCED"
        ? "Notes ready"
        : item.status === "SYNCING"
          ? item.syncProgress !== undefined
            ? `Processing… ${item.syncProgress}%`
            : "Syncing…"
          : "Saved offline";
    return (
      <View style={[styles.recordingCard, selected && styles.selectedCard]}>
        <View style={styles.recordingRow}>
          <View style={styles.recordingIcon}>
            <Text style={styles.recordingIconText}>♫</Text>
          </View>
          <Pressable
            style={styles.recordingInfo}
            onPress={() => setSelectedId(selected ? null : item.id)}
          >
            <Text style={styles.recordingTitle} numberOfLines={1}>{item.title}</Text>
            <Text style={styles.recordingMeta}>
              {formatDuration(item.durationSeconds)} · {new Date(item.createdAt).toLocaleDateString()}
            </Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={playingId === item.id ? "Pause audio" : "Play audio"}
            style={styles.playButton}
            onPress={() => void togglePlayback(item)}
          >
            <Text style={styles.playButtonText}>{playingId === item.id ? "Ⅱ" : "▶"}</Text>
          </Pressable>
        </View>
        <View style={styles.syncRow}>
          <View
            style={[
              styles.syncDot,
              item.status === "SYNCED" && styles.syncedDot,
              item.status === "SYNCING" && styles.syncingDot,
            ]}
          />
          <Text style={styles.syncLabel}>{statusLabel}</Text>
          {item.status === "SYNCING" && item.syncProgress !== undefined ? (
            <View style={styles.progressTrack}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${Math.max(0, Math.min(100, item.syncProgress))}%` },
                ]}
              />
            </View>
          ) : null}
          {item.status === "SYNCING" && item.syncMessage ? (
            <Text style={styles.progressText}>{item.syncMessage}</Text>
          ) : null}
          {item.status === "PENDING_SYNC" && isOnline ? (
            <Pressable
              style={styles.syncNow}
              onPress={() =>
                void syncPendingRecordings().then(refreshRecordings).catch(() => undefined)
              }
            >
              <Text style={styles.syncNowText}>Sync now</Text>
            </Pressable>
          ) : null}
          {item.status === "SYNCED" ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Re-process this recording"
              style={styles.syncNow}
              onPress={() => void reprocessRecording(item)}
            >
              <Text style={styles.syncNowText}>Re-process</Text>
            </Pressable>
          ) : null}
        </View>
        {item.lastError ? (
          <Text style={styles.errorText}>{item.lastError}</Text>
        ) : null}
        {selected && item.materials ? (
          <View style={styles.results}>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.tabList}
            >
              {tabs.map((tab) => (
                <Pressable
                  key={tab}
                  style={[styles.tab, selectedTab === tab && styles.activeTab]}
                  onPress={() => setSelectedTab(tab)}
                >
                  <Text style={[styles.tabText, selectedTab === tab && styles.activeTabText]}>
                    {tab}
                  </Text>
                </Pressable>
              ))}
            </ScrollView>
            {renderResults(item)}
          </View>
        ) : null}
        {selected && item.status === "SYNCING" ? (
          <View style={styles.loadingResult}>
            <ActivityIndicator color="#7664D8" />
            <Text style={styles.syncLabel}>Preparing your class notes…</Text>
          </View>
        ) : null}
        {selected && item.status === "PENDING_SYNC" ? (
          <Text style={styles.offlineHint}>
            Your recording is safe on this device. Notes will appear after sync.
          </Text>
        ) : null}
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <FlatList
        data={subjectRecordings}
        keyExtractor={(item) => item.id}
        renderItem={renderRecording}
        contentContainerStyle={styles.listContent}
        ListHeaderComponent={
          <View>
            <View style={styles.header}>
              <View>
                <Text style={styles.eyebrow}>YOUR PRIVATE STUDY SPACE</Text>
                <Text style={styles.title}>Class Notes</Text>
              </View>
              <View style={[styles.networkBadge, isOnline ? styles.onlineBadge : styles.offlineBadge]}>
                <View style={[styles.networkDot, isOnline && styles.networkDotOnline]} />
                <Text style={styles.networkText}>{isOnline ? "ONLINE" : "OFFLINE"}</Text>
              </View>
            </View>
            <Text style={styles.subtitle}>
              Choose a subject folder first. Recordings, dated notes, and quizzes stay organized in that folder.
            </Text>
            <View style={styles.folderSection}>
              <Text style={styles.folderHeading}>Subject folders</Text>
              {subjectFolders.length ? (
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  contentContainerStyle={styles.folderList}
                >
                  {subjectFolders.map((folder) => {
                    const selected = folder.id === selectedSubjectId;
                    return (
                      <Pressable
                        key={folder.id}
                        accessibilityRole="button"
                        accessibilityState={{ selected }}
                        disabled={isRecording}
                        onPress={() => selectSubject(folder)}
                        style={[
                          styles.folderChip,
                          selected && styles.selectedFolderChip,
                          isRecording && styles.disabledFolderChip,
                        ]}
                      >
                        <Text
                          style={[
                            styles.folderChipText,
                            selected && styles.selectedFolderChipText,
                          ]}
                        >
                          {folder.name}
                        </Text>
                      </Pressable>
                    );
                  })}
                </ScrollView>
              ) : (
                <Text style={styles.folderHint}>
                  Create a subject folder before starting your first recording.
                </Text>
              )}
              <View style={styles.folderCreateRow}>
                <TextInput
                  accessibilityLabel="New subject folder name"
                  value={newSubjectName}
                  onChangeText={setNewSubjectName}
                  placeholder="e.g. Biology"
                  placeholderTextColor="#AAA7B8"
                  maxLength={50}
                  editable={!isRecording}
                  onSubmitEditing={() => void addSubjectFolder()}
                  returnKeyType="done"
                  style={styles.folderInput}
                />
                <Pressable
                  accessibilityRole="button"
                  disabled={isRecording || !newSubjectName.trim()}
                  onPress={() => void addSubjectFolder()}
                  style={[
                    styles.createFolderButton,
                    (isRecording || !newSubjectName.trim()) && styles.createFolderButtonDisabled,
                  ]}
                >
                  <Text style={styles.createFolderButtonText}>Add folder</Text>
                </Pressable>
              </View>
            </View>
            {selectedSubject ? (
              <RecordingControls
                subjectId={selectedSubject.id}
                subjectName={selectedSubject.name}
                onRecordingStateChange={setIsRecording}
                onSaved={saveRecording}
              />
            ) : (
              <View style={styles.recordingPrompt}>
                <Text style={styles.recordingPromptText}>
                  Add and select a subject folder to enable recording.
                </Text>
              </View>
            )}
            <View style={styles.sectionHeader}>
              <View>
                <Text style={styles.sectionTitle}>
                  {selectedSubject ? `${selectedSubject.name} classes` : "Your classes"}
                </Text>
                <Text style={styles.sectionCaption}>
                  {subjectRecordings.filter((item) => item.status !== "SYNCED").length > 0
                    ? `${subjectRecordings.filter((item) => item.status !== "SYNCED").length} waiting to sync`
                    : "All notes are up to date"}
                </Text>
              </View>
              <Text style={styles.classCount}>{subjectRecordings.length}</Text>
            </View>
            {loadError ? (
              <Pressable style={styles.errorBanner} onPress={() => void refreshRecordings()}>
                <Text style={styles.errorText}>{loadError}  Tap to retry.</Text>
              </Pressable>
            ) : null}
            {isLoading ? (
              <ActivityIndicator style={styles.loading} color="#7664D8" />
            ) : null}
          </View>
        }
        ListEmptyComponent={
          !isLoading ? (
            <View style={styles.emptyState}>
              <View style={styles.emptyIcon}><Text style={styles.emptyIconText}>♫</Text></View>
              <Text style={styles.emptyTitle}>Your next great lesson starts here</Text>
              <Text style={styles.emptyText}>
                {selectedSubject
                  ? `Record a class in ${selectedSubject.name}. Its dated transcript, notes, and quiz will appear in this folder.`
                  : "Create a subject folder above to organize your classes, transcripts, notes, and quizzes."}
              </Text>
            </View>
          ) : null
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: "#F7F6FB" },
  listContent: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 44 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 24,
    marginBottom: 8,
  },
  eyebrow: {
    color: "#8A83B0",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1.4,
    marginBottom: 7,
  },
  title: { color: "#24213B", fontSize: 30, fontWeight: "800", letterSpacing: -0.8 },
  subtitle: {
    color: "#77758B",
    fontSize: 14,
    lineHeight: 21,
    marginBottom: 22,
    maxWidth: 300,
  },
  folderSection: {
    backgroundColor: "#FFFFFF",
    borderColor: "#E9E8F2",
    borderRadius: 18,
    borderWidth: 1,
    marginBottom: 15,
    padding: 14,
  },
  folderHeading: { color: "#38344E", fontSize: 14, fontWeight: "800" },
  folderList: { gap: 8, paddingVertical: 12 },
  folderChip: {
    backgroundColor: "#F5F4F8",
    borderColor: "#E9E6F2",
    borderRadius: 12,
    borderWidth: 1,
    paddingHorizontal: 13,
    paddingVertical: 9,
  },
  selectedFolderChip: { backgroundColor: "#7664D8", borderColor: "#7664D8" },
  disabledFolderChip: { opacity: 0.65 },
  folderChipText: { color: "#77758B", fontSize: 12, fontWeight: "700" },
  selectedFolderChipText: { color: "#FFFFFF" },
  folderHint: { color: "#89869A", fontSize: 12, lineHeight: 18, marginTop: 8 },
  folderCreateRow: { flexDirection: "row", gap: 8, marginTop: 4 },
  folderInput: {
    backgroundColor: "#F7F6FB",
    borderColor: "#E8E5F0",
    borderRadius: 11,
    borderWidth: 1,
    color: "#403B5C",
    flex: 1,
    fontSize: 13,
    paddingHorizontal: 11,
    paddingVertical: 9,
  },
  createFolderButton: {
    alignItems: "center",
    backgroundColor: "#7664D8",
    borderRadius: 11,
    justifyContent: "center",
    paddingHorizontal: 12,
  },
  createFolderButtonDisabled: { opacity: 0.5 },
  createFolderButtonText: { color: "#FFFFFF", fontSize: 11, fontWeight: "700" },
  recordingPrompt: {
    backgroundColor: "#EEEBFB",
    borderRadius: 14,
    marginBottom: 8,
    padding: 15,
  },
  recordingPromptText: { color: "#554C91", fontSize: 12, lineHeight: 18 },
  networkBadge: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 11,
    paddingVertical: 8,
    borderRadius: 20,
  },
  onlineBadge: { backgroundColor: "#E7F6EF" },
  offlineBadge: { backgroundColor: "#EFEDF5" },
  networkDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: "#9E9BAE", marginRight: 6 },
  networkDotOnline: { backgroundColor: "#32A875" },
  networkText: { color: "#69667A", fontSize: 10, fontWeight: "800", letterSpacing: 0.7 },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 30,
    marginBottom: 14,
  },
  sectionTitle: { color: "#24213B", fontSize: 19, fontWeight: "800" },
  sectionCaption: { color: "#89869A", fontSize: 12, marginTop: 4 },
  classCount: {
    color: "#6857C3",
    backgroundColor: "#EEEBFB",
    paddingHorizontal: 11,
    paddingVertical: 6,
    borderRadius: 12,
    fontSize: 12,
    fontWeight: "800",
    overflow: "hidden",
  },
  recordingCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 20,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#ECEAF3",
  },
  selectedCard: { borderColor: "#C6BDF2" },
  recordingRow: { flexDirection: "row", alignItems: "center" },
  recordingIcon: {
    height: 44,
    width: 44,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F0EDFC",
  },
  recordingIconText: { color: "#7664D8", fontSize: 24, fontWeight: "700" },
  recordingInfo: { flex: 1, marginHorizontal: 11 },
  recordingTitle: { color: "#302D47", fontSize: 14, fontWeight: "700" },
  recordingMeta: { color: "#9290A1", fontSize: 11, marginTop: 5 },
  playButton: {
    width: 37,
    height: 37,
    borderRadius: 13,
    backgroundColor: "#7664D8",
    alignItems: "center",
    justifyContent: "center",
  },
  playButtonText: { color: "#FFFFFF", fontSize: 14, fontWeight: "700" },
  syncRow: { flexDirection: "row", alignItems: "center", marginTop: 13 },
  syncDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: "#E6A63D", marginRight: 7 },
  syncedDot: { backgroundColor: "#36A978" },
  syncingDot: { backgroundColor: "#7664D8" },
  syncLabel: { color: "#888598", fontSize: 11, fontWeight: "600" },
  progressTrack: {
    flex: 1,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#E6E3F0",
    marginLeft: 8,
    overflow: "hidden",
  },
  progressFill: { height: 4, borderRadius: 2, backgroundColor: "#7664D8" },
  progressText: { color: "#A8A5B5", fontSize: 10, marginTop: 4 },
  syncNow: { marginLeft: "auto", padding: 4 },
  syncNowText: { color: "#6857C3", fontSize: 12, fontWeight: "700" },
  errorText: { color: "#BC5260", fontSize: 12, lineHeight: 18, marginTop: 8 },
  results: { marginTop: 15, borderTopWidth: 1, borderTopColor: "#F0EFF5" },
  tabList: { gap: 8, paddingTop: 13, paddingBottom: 4 },
  tab: {
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: "#F5F4F8",
  },
  activeTab: { backgroundColor: "#7664D8" },
  tabText: { color: "#77758B", fontSize: 11, fontWeight: "700" },
  activeTabText: { color: "#FFFFFF" },
  resultsContent: { paddingTop: 15, paddingBottom: 4 },
  transcriptHelp: { color: "#89869A", fontSize: 12, lineHeight: 18, marginBottom: 12 },
  speakerEditor: { marginBottom: 12 },
  speakerInput: {
    backgroundColor: "#F7F6FB",
    borderWidth: 1,
    borderColor: "#E8E5F0",
    borderRadius: 12,
    color: "#403B5C",
    fontSize: 13,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  renameButton: {
    alignSelf: "flex-start",
    backgroundColor: "#7664D8",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginTop: 7,
  },
  renameButtonDisabled: { opacity: 0.5 },
  renameButtonText: { color: "#FFFFFF", fontSize: 11, fontWeight: "700" },
  forgetButton: { alignSelf: "flex-start", paddingHorizontal: 2, paddingVertical: 8 },
  forgetButtonText: { color: "#BC5260", fontSize: 11, fontWeight: "700" },
  transcriptTurn: {
    borderTopWidth: 1,
    borderTopColor: "#F0EFF5",
    paddingVertical: 10,
  },
  transcriptTurnHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 4,
  },
  transcriptSpeaker: { color: "#6857C3", fontSize: 12, fontWeight: "800" },
  transcriptTime: { color: "#A09DAD", fontSize: 10, fontVariant: ["tabular-nums"] },
  transcriptText: { color: "#555267", fontSize: 13, lineHeight: 20 },
  interruptionLabel: { color: "#A16A2E", fontSize: 10, fontWeight: "700", marginTop: 4 },
  resultHeading: { color: "#38344E", fontSize: 13, fontWeight: "800", marginTop: 12, marginBottom: 7 },
  bullet: { color: "#737185", fontSize: 12, lineHeight: 19, marginBottom: 5 },
  qaItem: { marginBottom: 10 },
  qaQuestion: { color: "#403B5C", fontSize: 12, fontWeight: "700", lineHeight: 18 },
  quizItem: { borderBottomWidth: 1, borderBottomColor: "#F0EFF5", paddingBottom: 10, marginBottom: 8 },
  answer: { color: "#468867", fontSize: 11, fontWeight: "700", marginTop: 3 },
  loadingResult: { flexDirection: "row", alignItems: "center", gap: 9, paddingVertical: 20 },
  offlineHint: { color: "#89869A", fontSize: 12, lineHeight: 18, marginTop: 14 },
  emptyState: {
    alignItems: "center",
    backgroundColor: "#FFFFFF",
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "#ECEAF3",
    paddingHorizontal: 24,
    paddingVertical: 27,
  },
  emptyIcon: {
    width: 50,
    height: 50,
    borderRadius: 17,
    backgroundColor: "#F0EDFC",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 13,
  },
  emptyIconText: { color: "#7664D8", fontSize: 27 },
  emptyTitle: { color: "#393550", fontSize: 14, fontWeight: "800", textAlign: "center" },
  emptyText: { color: "#89869A", fontSize: 12, lineHeight: 18, textAlign: "center", marginTop: 7 },
  loading: { padding: 20 },
  errorBanner: { backgroundColor: "#FBECEE", borderRadius: 12, padding: 12 },
});
