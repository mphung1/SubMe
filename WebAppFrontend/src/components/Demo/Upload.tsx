import { useState, useEffect } from "react";
import { Center, Grid } from "@chakra-ui/react";
import CustomButton from "@components/Fixed/CustomButton"

const { createFFmpeg, fetchFile, renderContentScreen } = require("@ffmpeg/ffmpeg");
const ffmpeg = createFFmpeg({ log: true });
const validFileTypes = "video/*, audio/*";

const Upload = ({ uploadCallback, uploadTypeCallback, renderContentScreen }) => {
  const [loaded, setLoaded] = useState(false);
  const [video, setVideo] = useState<any | undefined>();

  const fileMaxSize = 1000000000;

  const load = async () => {
    await ffmpeg.isLoaded();
    setLoaded(true);
  };

  useEffect(() => {
    load();
  }, []);

  return loaded ? (
    <>
      <Center mt="2rem">
        <input
          type="file"
          accept={validFileTypes}
          onChange={(event) => {
            if (event.target.files?.item(0).size < fileMaxSize) {
              event.preventDefault();
              setVideo(event.target.files?.item(0));
              uploadCallback(URL.createObjectURL(event.target.files?.item(0)));
              uploadTypeCallback(event.target.files?.item(0));
            } else {
              window.alert(
                "Your file cannot be larger than " +
                  Number(fileMaxSize / 1000000000) +
                  "GB."
              );
            }
          }}
        />
      </Center>
      {video && (
        <div>
          <Center mt="1rem">
          <Grid templateColumns='repeat(1, 1fr)' gap={2}>
            <video
              controls
              width="500"
              height="250"
              src={URL.createObjectURL(video)}
            />
            <CustomButton btnText="Proceed" onClick={renderContentScreen} />
          </Grid>
          </Center>
        </div>
      )}
    </>
  ) : (
    <p>Loading ...</p>
  );
};

export default Upload;
