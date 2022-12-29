import { useState } from "react";
import {
  Container,
  Text,
  Box,
  Input,
  UnorderedList,
  ListItem,
  Grid
} from "@chakra-ui/react";
import ReactPlayer from "react-player";
import CustomButton from "@components/Fixed/CustomButton";

function UrlReader({ urlCallback, renderContentScreen }) {
  const [value, setValue] = useState<string>();
  const [url, setUrl] = useState<string>();
  const [message, showMessage] = useState<boolean>(true);

  const handleChange = (event) => setValue(event.target.value);
  const handleSubmit = (event) => {
    event.preventDefault();
    setUrl(value);
    showMessage(false);
    urlCallback(value);
  };

  return (
    <>
      <Container mt="1.5rem">
        <Text mb="8px">Paste Youtube URL here</Text>
        <form onSubmit={handleSubmit}>
          <Input
            onChange={handleChange}
            placeholder="https://"
            size="sm"
            focusBorderColor="blue.500"
          />
        </form>
        {message ? (
          <UnorderedList>
            <ListItem color="gray" fontSize="md" mt="6rem">
              We accept a variety of URLs, including file paths, YouTube,
              Facebook, Twitch, SoundCloud, Streamable, Vimeo, Wistia, Mixcloud,
              DailyMotion and Kaltura.
            </ListItem>
            <ListItem color="gray" fontSize="md">
              Click enter. If your URL is correct, you&apos;ll see a video
              preview here.
            </ListItem>
          </UnorderedList>
        ) : (
          <Box pos="relative" padding-top="56.25%" mt="1rem">
            <Grid templateColumns='repeat(1, 1fr)' gap={2}>
              <ReactPlayer url={url} controls={true} width="100%" height={300} />
              <CustomButton btnText="Proceed" onClick={renderContentScreen}/>
            </Grid>
          </Box>
        )}
      </Container>
    </>
  );
}

export default UrlReader;
