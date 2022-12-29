import { useState, useEffect, useReducer } from "react";
import transcription from "@pages/api/transcription";
import ClimbingBoxLoader from "react-spinners/ClimbingBoxLoader";
import {
  Container,
  Box,
  Flex,
  Text,
  Heading,
  Divider,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Center,
  Spacer,
  VStack,
  Button
} from "@chakra-ui/react";
import ReactPlayer from "react-player";
import ModelSelect from "react-select";
import CustomButton from "@components/Fixed/CustomButton";
import Search from "@components/Demo/Search";
import Upload from "@components/Demo/Upload";
import UrlReader from "@components/Demo/UrlReader";
import InfoPopOver from "@components/Demo/InfoPopOver";

export default function Test() {
  const modelOptions = [
    {
      value: "quartznet",
      label: "QuartzNet",
    },
    {
      value: "jasper",
      label: "Jasper",
    },
  ];

  const [loading, setLoading] = useState(false);
  const [url, setUrl] = useState<string>();
  const [options, showOptions] = useState(true);
  const [model, setModel] = useState<string>();
  const [inputFileType, setInputFileType] = useState<string>();
  const [uploadedFile, handleUploadedFile] = useState<Blob | string | string[]>(
    null
  );
  const [transcript, showTranscript] = useState<string>();
  const [latencyNumber, showLatencyNumber] = useState<number>();

  const FormData = require("form-data");
  const bodyFormData = new FormData();
  bodyFormData.append("dataset_name", "LibriSpeech");
  bodyFormData.append("model_name", `${model}`);
  bodyFormData.append("file_datatype", `${inputFileType}`);
  if (uploadedFile != null) {
    bodyFormData.append("file", uploadedFile);
  }

  const config = {
    headers: {
      "Access-Control-Allow-Origin": "*/*",
      "Content-Type": "multipart/form-data",
      mode: "no-cors",
    },
  };

  const handleTranscribe = async () => {
    setLoading(true);
    const res = await transcription.post(
      "/transcribe_file",
      bodyFormData,
      config
    );
    showTranscript(res.data.transcript);
    showLatencyNumber(res.data.latency);
    setLoading(false);
  };

  const setSelectedModel = (event: { value: React.SetStateAction<string> }) => {
    setModel(event.value);
  };

  const handleSearchCallback = (videoId: string) => {
    setUrl("https://www.youtube.com/embed/" + videoId);
    handleUploadedFile("https://www.youtube.com/embed/" + videoId);
    setInputFileType("y_video");
  };

  const handleUploadCallback = (fileUrl: string) => {
    setUrl(fileUrl);
  };

  const handleUrlCallback = (watchUrl: string) => {
    setUrl(watchUrl);
    handleUploadedFile(watchUrl);
    setInputFileType("y_video");
  };

  const handleUploadTypeCallback = (blob: Blob) => {
    handleUploadedFile(blob);
    if (blob.type == "audio/wav") {
      setInputFileType("a_upload");
    } else if (blob.type == "video/mp4") {
      setInputFileType("v_upload");
    } else {
      setInputFileType("not_supported");
      alert("Unsupported format!");
    }
  };

  useEffect(() => {
    setLoading(false);
  }, [uploadedFile]);

  const [volume, setVolume] = useReducer(reducer, 0.4);
  function reducer(state: number, action: { type: string }) {
    switch (action.type) {
      case "increment":
        return state + 0.1;
      case "decrement":
        return state - 0.1;
      default:
        throw new Error();
    }
  }

  const renderContentScreen = () => {
    showOptions(false);
  };

  const renderOptionScreen = () => {
    showOptions(true);
  };

  return (
    <>
      {options ? (
        <>
          <Tabs isFitted>
            <TabList mb="1rem">
              <Tab _selected={{ color: "white", bg: "black" }}>Search</Tab>
              <Tab _selected={{ color: "white", bg: "black" }}>Upload</Tab>
              <Tab _selected={{ color: "white", bg: "black" }}>By URL</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <Search searchCallback={handleSearchCallback} renderContentScreen={renderContentScreen}/>
              </TabPanel>
              <TabPanel>
                <Upload
                  uploadCallback={handleUploadCallback}
                  uploadTypeCallback={handleUploadTypeCallback}
                  renderContentScreen={renderContentScreen}
                />
              </TabPanel>
              <TabPanel>
                <UrlReader urlCallback={handleUrlCallback} renderContentScreen={renderContentScreen}/>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </>
      ) : (
        <>
          <Flex
            maxW="container.xl"
            bg="white"
            style={{
              display: "flex",
              flexDirection: "row",
              borderColor: "black",
            }}
          >
            <Box
              pos="relative"
              text-align="left"
              display="inline-block"
              style={{ borderWidth:"0.1rem", borderColor:"black"}}
            >
              <VStack spacing={2} align="stretch">
                <CustomButton
                  btnText="Go back"
                  variant="ghost"
                  onClick={renderOptionScreen}
                />
                <ReactPlayer
                  position="absolute"
                  key={url}
                  url={url}
                  width="100%"
                  height={300}
                  controls={true}
                  playing={false}
                  volume={volume}
                />
                <Box ml={2}>
                  <ModelSelect
                    placeholder="Choose model"
                    options={modelOptions}
                    onChange={setSelectedModel}
                  />
                </Box>
                <CustomButton btnText="Transcribe" onClick={handleTranscribe} />
                <CustomButton
                  btnText="Transcribe with Timestamp"
                  onClick={() =>
                    window.alert("This feature is under development.")
                  }
                />
                <CustomButton btnText="Export" ml={1} />

              </VStack>
            </Box>
            <Box
              pos="relative"
              h="100%"
              w="100%"
              padding="1rem"
              bg="white"
              color="black"
              minW="container.md"
              minH="xl"
              display="inline-block"
              style={{ borderWidth:"0.1rem", borderColor:"black"}}
            >
              <Flex align="flex-end">
                <Heading size="lg" letterSpacing="tight">
                  Transcription
                </Heading>
                <Spacer />
                <Heading size="sm" mr="7rem">
                  Latency:{" "}
                </Heading>
                {latencyNumber && <Text>{latencyNumber}s</Text>}
              </Flex>

              <Divider orientation="horizontal" variant="solid" />
              <Flex align="flex-end">
                <table>
                  <tbody>
                    <tr>
                      <th>Dataset Name: </th>
                      <Text as="i" ml={2}>
                        LibriSpeech
                      </Text>
                    </tr>
                    <tr>
                      <th>Model Name:</th>
                      <Text as="i">{model}</Text>
                    </tr>
                    <tr>
                      <th>Input Type:</th>
                      <Text as="i">{inputFileType}</Text>
                    </tr>
                    <tr>
                      <th>File URL:</th>
                      <Text as="i">{url}</Text>
                    </tr>
                  </tbody>
                </table>
              </Flex>
              <Divider orientation="horizontal" variant="solid" />

              {transcript && <Text>{transcript}</Text>}
              <Center mt="3rem">
                <ClimbingBoxLoader loading={loading} />
              </Center>
            </Box>
          </Flex>
        </>
      )}
    </>
  );
}
